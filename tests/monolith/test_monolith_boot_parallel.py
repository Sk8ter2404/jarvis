"""v3 W10 - BOOT_PARALLEL_LOAD: the dependency check in two passes and the
boot Whisper load off the boot thread when Parakeet is primary.

  * the fast dependency pass (find_spec, no import) still names - and speaks,
    in the old words - a package that is not installed;
  * the background full import check reports only what the fast pass missed,
    through the proactive queue;
  * a Parakeet -> Whisper rescue while the background load runs is skipped
    ("no rescue") and never raises;
  * the boot thread starts the CUDA driver before the Whisper thread exists
    (v2.0.180: before ANY ctranslate2 import), and nothing on the boot path
    of the fast pass imports faster_whisper / ctranslate2;
  * with the setting off, main() takes the old calls exactly.

Everything external is mocked; no model loads, no audio, no network.
"""
from __future__ import annotations

import ast
import contextlib
import inspect
import io
import logging

import textwrap
import threading
import unittest
from unittest import mock

from tests._monolith_harness import (
    MonolithGlobalsTestCase, load_monolith, requires_monolith)

_STUB = "jarvis-w10-no-such-package"


def _calls(fn_src: str):
    tree = ast.parse(textwrap.dedent(fn_src))
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            f = n.func
            name = f.id if isinstance(f, ast.Name) else (
                f.attr if isinstance(f, ast.Attribute) else "")
            out.append((n.lineno, name))
    return sorted(out)


class _Base(MonolithGlobalsTestCase):
    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def setUp(self):
        self._saved_bg = dict(self.bc._whisper_bg)
        self._saved_dep_thread = self.bc._dep_full_thread[0]
        self.bc._dep_full_thread[0] = None
        self._saved_setting = self.bc.BOOT_PARALLEL_LOAD
        self._fw_logger = logging.getLogger("faster_whisper")
        self._fw_level = self._fw_logger.level

    def tearDown(self):
        self.bc._whisper_bg.clear()
        self.bc._whisper_bg.update(self._saved_bg)
        self.bc._dep_full_thread[0] = self._saved_dep_thread
        self.bc.BOOT_PARALLEL_LOAD = self._saved_setting
        self._fw_logger.setLevel(self._fw_level)


@requires_monolith
class SettingTests(_Base):
    def test_default_is_off_and_declared(self):
        import core.config as cfg
        self.assertIs(cfg.BOOT_PARALLEL_LOAD, False)
        self.assertIn("BOOT_PARALLEL_LOAD", self.bc._TURN_FLAG_KEYS)

    def test_reader(self):
        bc = self.bc
        for v, want in ((False, False), (True, True), ("on", True),
                        ("true", True), ("off", False), ("", False),
                        (None, False), (1, False)):
            bc.BOOT_PARALLEL_LOAD = v
            self.assertIs(bc._boot_parallel_load_on(), want, repr(v))


@requires_monolith
class FastDependencyPassTests(_Base):
    def test_fast_pass_reports_a_missing_stub_package_in_the_old_words(self):
        import importlib
        speak = mock.Mock()
        hist = []
        out = io.StringIO()
        with mock.patch.object(self.bc, "_parse_requirements",
                               return_value=["numpy", _STUB]), \
                mock.patch.object(self.bc, "_DEP_FEATURE_NOTE",
                                  {_STUB: "the stub feature is offline"}), \
                mock.patch.object(importlib, "import_module",
                                  side_effect=AssertionError("imported")), \
                mock.patch.object(self.bc, "_speak", speak), \
                mock.patch.object(self.bc, "conversation_history", hist), \
                contextlib.redirect_stdout(out):
            missing = self.bc.check_dependencies_fast()
        self.assertEqual(missing, [_STUB])
        speak.assert_called_once_with(
            f"Sir, the stub feature is offline — {_STUB} is missing.")
        self.assertEqual(hist, [{"role": "assistant", "content":
                                 speak.call_args[0][0]}])
        text = out.getvalue()
        self.assertIn("[dep-check] WARNING: 1 package(s)", text)
        self.assertIn(f"pip install {_STUB}", text)

    def test_fast_pass_wording_matches_the_full_check(self):
        """Same stub, same notes: the two report the same lines and alert."""
        notes = {"pkgA": "alpha", "pkgB": "beta", "pkgC": "gamma"}
        spoken = []

        def run(fn):
            out = io.StringIO()
            with mock.patch.object(self.bc, "_parse_requirements",
                                   return_value=["pkgA", "pkgB", "pkgC"]), \
                    mock.patch.object(self.bc, "_DEP_IMPORT_NAME", {}), \
                    mock.patch.object(self.bc, "_DEP_FEATURE_NOTE", notes), \
                    mock.patch.object(self.bc, "_speak", spoken.append), \
                    mock.patch.object(self.bc, "conversation_history", []), \
                    contextlib.redirect_stdout(out):
                fn()
            return out.getvalue()

        a = run(self.bc.check_dependencies)
        b = run(self.bc.check_dependencies_fast)
        self.assertEqual(a, b)
        self.assertEqual(len(spoken), 2)
        self.assertEqual(spoken[0], spoken[1])
        self.assertIn("and 1 more", spoken[0])

    def test_fast_pass_all_found(self):
        out = io.StringIO()
        with mock.patch.object(self.bc, "_parse_requirements",
                               return_value=["numpy", "requests"]), \
                contextlib.redirect_stdout(out):
            self.assertEqual(self.bc.check_dependencies_fast(), [])
        self.assertIn("[dep-check] fast pass: all requirements.txt packages "
                      "found", out.getvalue())

    def test_find_spec_error_counts_as_missing_and_never_raises(self):
        import importlib.util as ilu
        with mock.patch.object(ilu, "find_spec",
                               side_effect=ValueError("bad spec")):
            self.assertEqual(self.bc._dep_missing_fast(["numpy"]), ["numpy"])

    def test_fast_pass_imports_nothing(self):
        for name in ("_dep_missing_fast", "check_dependencies_fast"):
            src = inspect.getsource(getattr(self.bc, name))
            self.assertNotIn("import_module", src, name)
            self.assertNotIn("faster_whisper", src, name)
            self.assertNotIn("ctranslate2", src, name)


@requires_monolith
class FullBackgroundCheckTests(_Base):
    def _run_worker(self, pkgs, already, broken=()):
        import importlib
        real = importlib.import_module

        def fake(name, *a, **k):
            if name in broken:
                raise ImportError(name)
            return real(name, *a, **k)

        announce = mock.Mock(return_value=True)
        speak = mock.Mock()
        out = io.StringIO()
        with mock.patch.object(self.bc, "_parse_requirements",
                               return_value=list(pkgs)), \
                mock.patch.object(self.bc, "_DEP_IMPORT_NAME", {}), \
                mock.patch.object(self.bc, "_DEP_FEATURE_NOTE",
                                  {"pkgA": "alpha", "pkgB": "beta"}), \
                mock.patch.object(importlib, "import_module",
                                  side_effect=fake), \
                mock.patch.object(self.bc, "proactive_announce", announce), \
                mock.patch.object(self.bc, "_speak", speak), \
                contextlib.redirect_stdout(out):
            self.bc._dep_check_full_worker(already)
        speak.assert_not_called()
        return announce, out.getvalue()

    def test_reports_only_what_the_fast_pass_missed(self):
        announce, text = self._run_worker(
            ["numpy", "pkgA", "pkgB"], already=["pkgA"],
            broken=("pkgA", "pkgB"))
        announce.assert_called_once_with(
            "Sir, beta — pkgB is missing.", source="dep-check")
        self.assertIn("[dep-check] WARNING: 1 package(s)", text)
        self.assertNotIn("- pkgA", text)

    def test_nothing_new_is_silent(self):
        announce, text = self._run_worker(
            ["numpy", "pkgA"], already=["pkgA"], broken=("pkgA",))
        announce.assert_not_called()
        self.assertEqual(text, "")

    def test_all_importable_prints_the_old_line(self):
        announce, text = self._run_worker(["numpy"], already=[])
        announce.assert_not_called()
        self.assertIn("[dep-check] all requirements.txt packages importable.",
                      text)

    def test_worker_never_raises(self):
        out = io.StringIO()
        with mock.patch.object(self.bc, "_parse_requirements",
                               side_effect=RuntimeError("boom")), \
                contextlib.redirect_stdout(out):
            self.bc._dep_check_full_worker([])
        self.assertIn("background check failed", out.getvalue())

    def test_start_runs_once_on_a_daemon_thread(self):
        ran = threading.Event()
        with mock.patch.object(self.bc, "_dep_check_full_worker",
                               side_effect=lambda a: ran.set()):
            t = self.bc._dep_check_full_start(["x"])
            self.assertIsNotNone(t)
            self.assertTrue(t.daemon)
            t.join(5.0)
            self.assertIs(self.bc._dep_check_full_start(["x"]), t)
        self.assertTrue(ran.is_set())


@requires_monolith
class RescueWhileLoadingTests(_Base):
    def test_rescue_is_skipped_while_whisper_loads(self):
        bc = self.bc
        bc._whisper_bg["loading"] = True
        out = io.StringIO()
        with mock.patch.object(bc._stt_parakeet, "rescue_reason",
                               return_value="empty"), \
                contextlib.redirect_stdout(out):
            self.assertEqual(bc._parakeet_rescue("", None), "")
        self.assertIn("[stt] rescue skipped (whisper loading)",
                      out.getvalue())

    def test_rescue_unchanged_when_not_loading(self):
        bc = self.bc
        bc._whisper_bg["loading"] = False
        out = io.StringIO()
        with mock.patch.object(bc._stt_parakeet, "rescue_reason",
                               return_value="no-wake"), \
                contextlib.redirect_stdout(out):
            self.assertEqual(bc._parakeet_rescue("hello", None), "no-wake")
        self.assertEqual(out.getvalue(), "")

    def test_no_reason_prints_nothing_while_loading(self):
        bc = self.bc
        bc._whisper_bg["loading"] = True
        out = io.StringIO()
        with mock.patch.object(bc._stt_parakeet, "rescue_reason",
                               return_value=""), \
                contextlib.redirect_stdout(out):
            self.assertEqual(bc._parakeet_rescue("jarvis hi", None), "")
        self.assertEqual(out.getvalue(), "")

    def test_primary_keeps_parakeet_text_and_never_calls_whisper(self):
        bc = self.bc
        bc._whisper_bg["loading"] = True
        whisper = mock.Mock(side_effect=AssertionError("whisper called"))
        latch = bc._stt_parakeet.Latch()
        p = bc._stt_parakeet.Primary(
            lambda a: ("", {"avg_logprob": -0.2}), whisper, latch=latch,
            rescue=lambda t, a: bc._parakeet_rescue(t, a), log=lambda s: None)
        out = io.StringIO()
        with mock.patch.object(bc._stt_parakeet, "rescue_reason",
                               return_value="empty"), \
                contextlib.redirect_stdout(out):
            text, conf = p.run(object())
        self.assertEqual(text, "")
        whisper.assert_not_called()
        self.assertEqual(p.rescues, 0)
        self.assertFalse(latch.failed)


@requires_monolith
class BootWhisperStartTests(_Base):
    def test_off_returns_false_and_touches_nothing(self):
        bc = self.bc
        bc.BOOT_PARALLEL_LOAD = False
        self._fw_logger.setLevel(logging.NOTSET)
        with mock.patch.object(bc, "_cuda_driver_first") as drv, \
                mock.patch.object(bc.threading, "Thread") as th:
            self.assertFalse(bc._boot_whisper_start())
        drv.assert_not_called()
        th.assert_not_called()
        self.assertFalse(bc._whisper_bg_loading())
        self.assertEqual(self._fw_logger.level, logging.NOTSET)

    def test_on_with_whisper_primary_stays_synchronous(self):
        bc = self.bc
        bc.BOOT_PARALLEL_LOAD = True
        with mock.patch.object(bc, "_stt_r6_route", return_value=None), \
                mock.patch.object(bc.threading, "Thread") as th:
            self.assertFalse(bc._boot_whisper_start())
        th.assert_not_called()
        self.assertEqual(self._fw_logger.level, logging.WARNING)

    def test_on_with_parakeet_loads_in_background_driver_first(self):
        bc = self.bc
        bc.BOOT_PARALLEL_LOAD = True
        order = []
        gate = threading.Event()

        def fake_ensure():
            order.append(("ensure", threading.current_thread().name,
                          bc._whisper_bg_loading()))
            gate.wait(5.0)

        real_thread = threading.Thread

        def spy_thread(*a, **k):
            order.append(("thread",))
            return real_thread(*a, **k)

        out = io.StringIO()
        with mock.patch.object(bc, "_stt_r6_route", return_value="primary"), \
                mock.patch.object(bc, "_stt", None), \
                mock.patch.object(bc, "_cuda_driver_first",
                                  side_effect=lambda: order.append(
                                      ("driver",
                                       threading.current_thread().name))), \
                mock.patch.object(bc, "_ensure_whisper",
                                  side_effect=fake_ensure), \
                mock.patch.object(bc.threading, "Thread",
                                  side_effect=spy_thread), \
                contextlib.redirect_stdout(out):
            self.assertTrue(bc._boot_whisper_start())
            self.assertTrue(bc._whisper_bg_loading())
            gate.set()
            bc._whisper_bg["thread"].join(5.0)
        self.assertFalse(bc._whisper_bg_loading())
        self.assertEqual(order[0], ("driver", threading.current_thread().name))
        self.assertEqual(order[1], ("thread",))
        self.assertEqual(order[2][0], "ensure")
        self.assertEqual(order[2][1], "boot-whisper")
        self.assertTrue(order[2][2])

    def test_worker_failure_clears_the_flag_and_never_raises(self):
        bc = self.bc
        bc._whisper_bg["loading"] = True
        out = io.StringIO()
        with mock.patch.object(bc, "_ensure_whisper",
                               side_effect=RuntimeError("cuda gone")), \
                contextlib.redirect_stdout(out):
            bc._boot_whisper_worker()
        self.assertFalse(bc._whisper_bg_loading())
        self.assertIn("background whisper load failed", out.getvalue())

    def test_quiet_logger(self):
        self._fw_logger.setLevel(logging.INFO)
        self.bc._quiet_faster_whisper_log()
        self.assertEqual(self._fw_logger.level, logging.WARNING)


@requires_monolith
class BootOrderStaticTests(_Base):
    def test_driver_starts_before_the_whisper_thread(self):
        calls = _calls(inspect.getsource(self.bc._boot_whisper_start))
        names = [n for _, n in calls]
        self.assertIn("_cuda_driver_first", names)
        self.assertIn("Thread", names)
        drv = min(ln for ln, n in calls if n == "_cuda_driver_first")
        thr = min(ln for ln, n in calls if n == "Thread")
        self.assertLess(drv, thr)

    def test_no_ct2_import_on_the_boot_thread_before_the_driver(self):
        """main(): the driver starts before the background-load branch, and
        neither the branch helper nor the fast dependency pass import
        faster_whisper / ctranslate2 themselves."""
        src = inspect.getsource(self.bc.main)
        drv = src.index("_cuda_driver_first()")
        self.assertLess(drv, src.index("if not _boot_whisper_start():"))
        self.assertLess(src.index("if not _boot_whisper_start():"),
                        src.index("_ensure_whisper()   # load now"))
        for name in ("_boot_whisper_start", "_boot_parallel_load_on",
                     "_quiet_faster_whisper_log", "check_dependencies_fast",
                     "_dep_missing_fast"):
            tree = ast.parse(textwrap.dedent(
                inspect.getsource(getattr(self.bc, name))))
            for n in ast.walk(tree):
                if isinstance(n, ast.Import):
                    mods = [a.name.split(".")[0] for a in n.names]
                elif isinstance(n, ast.ImportFrom):
                    mods = [(n.module or "").split(".")[0]]
                else:
                    continue
                self.assertFalse({"faster_whisper", "ctranslate2"} & set(mods),
                                 f"{name} imports {mods}")

    def test_off_path_keeps_the_old_calls(self):
        src = inspect.getsource(self.bc.main)
        self.assertIn("        _ensure_whisper()   # load now so first user "
                      "utterance isn't delayed", src)
        self.assertIn("            _missing_deps = check_dependencies()", src)
        # The full check starts once boot is done, just ahead of the
        # listening-mode build, which stays the statement right before the
        # main loop (test_monolith_listen_media_review).
        self.assertRegex(
            src.replace("\r\n", "\n"),
            r"        if _dep_parallel:\n"
            r"            _dep_check_full_start\(_missing_deps\)\n"
            r"(?:        #[^\n]*\n)*"
            r"        _listen_media_boot\(\)\n"
            r"        while True:")


if __name__ == "__main__":
    unittest.main()
