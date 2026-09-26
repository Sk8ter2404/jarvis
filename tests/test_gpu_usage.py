"""Logic tests for core.gpu_usage — the live GPU/VRAM usage engine.

Every data source is mocked end-to-end so NO real nvidia-smi runs, no Ollama
HTTP request leaves the box, and no `ollama ps` subprocess spawns:
  • Ollama /api/ps        → mock urllib.request.urlopen with a canned JSON body
  • `ollama ps` fallback  → mock subprocess.run returning the canned table text
  • nvidia-smi            → mock subprocess.run returning the canned CSV line

The contract under test:
  • gpu_snapshot() parses /api/ps JSON (size_vram bytes → MB) AND the nvidia-smi
    CSV (used/free/total/util/temp); falls back to the `ollama ps` table when
    the HTTP endpoint is down; degrades to a partial dict when a source is
    missing; NEVER raises.
  • usage_lines() / usage_bar() / routing_line() format the per-model rows, the
    TOTAL line (with percent + util + temp), the routing line, and the text bar
    exactly, and tolerate missing pieces.

stdlib unittest + unittest.mock only.
"""
from __future__ import annotations

import json
import unittest
from unittest import mock

from core import gpu_usage


# ─── canned source payloads (mirror the live formats verified on the 3090) ──

# GET /api/ps body: two LLMs + the embedder, all fully GPU-resident.
_API_PS_BODY = {
    "models": [
        {"name": "qwen2.5:14b-instruct-q5_K_M",
         "model": "qwen2.5:14b-instruct-q5_K_M",
         "size": 13455222373, "size_vram": 13455222373,
         "context_length": 16384},
        {"name": "qwen2.5vl:7b", "model": "qwen2.5vl:7b",
         "size": 7332597595, "size_vram": 7332597595,
         "context_length": 32768},
        {"name": "nomic-embed-text:latest", "model": "nomic-embed-text:latest",
         "size": 323150151, "size_vram": 323150151,
         "context_length": 2048},
    ]
}

# `ollama ps` table — the fallback when /api/ps is unreachable.
_OLLAMA_PS_TEXT = (
    "NAME                           ID              SIZE      PROCESSOR    CONTEXT    UNTIL\n"
    "qwen2.5:14b-instruct-q5_K_M    7bb3f324cafc    13 GB     100% GPU     16384      19 minutes from now\n"
    "qwen2.5vl:7b                   5ced39dfa4ba    7.3 GB    100% GPU     32768      4 minutes from now\n"
    "nomic-embed-text:latest        0a109f422b47    323 MB    100% GPU     2048       4 minutes from now\n"
)

# A model that's spilled partly onto the CPU (PROCESSOR shows a split).
_OLLAMA_PS_SPILL = (
    "NAME            ID              SIZE      PROCESSOR        CONTEXT    UNTIL\n"
    "qwen2.5:32b     abc123def456    21 GB     12%/88% CPU/GPU  12288      30 minutes from now\n"
)

# nvidia-smi CSV: used, free, total, util%, temp°C (single 24 GB card).
_SMI_LINE = "23722, 605, 24576, 39, 33\n"
# Two-GPU CSV — memory should sum, util/temp should take the max.
_SMI_MULTI = "10000, 2000, 12000, 20, 40\n8000, 4000, 12000, 55, 61\n"


# ─── helpers to stub each source ────────────────────────────────────────────

def _fake_urlopen(body: dict | bytes | Exception):
    """Build an object usable as `urllib.request.urlopen`'s return (a context
    manager whose .read() yields the JSON bytes), or a side_effect raiser."""
    if isinstance(body, Exception):
        return mock.MagicMock(side_effect=body)
    raw = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
    cm = mock.MagicMock()
    cm.__enter__.return_value.read.return_value = raw
    cm.__exit__.return_value = False
    return mock.MagicMock(return_value=cm)


def _smi_ok(stdout=_SMI_LINE):
    return mock.MagicMock(returncode=0, stdout=stdout)


def _ps_ok(stdout=_OLLAMA_PS_TEXT):
    return mock.MagicMock(returncode=0, stdout=stdout)


class _GpuUsageBase(unittest.TestCase):
    def setUp(self):
        # Bust the module-level snapshot cache before every test so a prior
        # test's snapshot can't satisfy `use_cache=True`.
        with gpu_usage._cache_lock:
            gpu_usage._cache = None
            gpu_usage._cache_at = 0.0
        self.addCleanup(self._bust)

    def _bust(self):
        with gpu_usage._cache_lock:
            gpu_usage._cache = None
            gpu_usage._cache_at = 0.0


# ─── /api/ps parsing ────────────────────────────────────────────────────────

class ApiPsTests(_GpuUsageBase):
    def test_models_from_api_shape_and_mb(self):
        rows = gpu_usage._models_from_api(_API_PS_BODY["models"])
        self.assertEqual([r["name"] for r in rows],
                         ["qwen2.5:14b-instruct-q5_K_M", "qwen2.5vl:7b",
                          "nomic-embed-text:latest"])
        # 13455222373 bytes ≈ 12832 MB.
        self.assertEqual(rows[0]["vram_mb"], 12832)
        self.assertEqual(rows[1]["vram_mb"], 6993)
        self.assertEqual(rows[2]["vram_mb"], 308)
        self.assertTrue(all(r["processor"] == "100% GPU" for r in rows))

    def test_fetch_returns_models_list(self):
        with mock.patch.object(gpu_usage.urllib.request, "urlopen",
                               _fake_urlopen(_API_PS_BODY)):
            models = gpu_usage._fetch_ollama_ps_json()
        self.assertIsInstance(models, list)
        self.assertEqual(len(models), 3)

    def test_fetch_none_on_http_error(self):
        with mock.patch.object(gpu_usage.urllib.request, "urlopen",
                               side_effect=OSError("conn refused")):
            self.assertIsNone(gpu_usage._fetch_ollama_ps_json())

    def test_fetch_none_on_bad_json(self):
        with mock.patch.object(gpu_usage.urllib.request, "urlopen",
                               _fake_urlopen(b"not json{{")):
            self.assertIsNone(gpu_usage._fetch_ollama_ps_json())

    def test_size_vram_missing_falls_back_to_size(self):
        rows = gpu_usage._models_from_api([
            {"name": "m1", "size": 2 * 1024 * 1024 * 1024},  # only `size`
        ])
        self.assertEqual(rows[0]["vram_mb"], 2048)
        self.assertEqual(rows[0]["processor"], "100% GPU")

    def test_partial_gpu_processor_label(self):
        # 88% on GPU → "12%/88% CPU/GPU".
        rows = gpu_usage._models_from_api([
            {"name": "m1", "size": 1000, "size_vram": 880},
        ])
        self.assertEqual(rows[0]["processor"], "12%/88% CPU/GPU")

    def test_blank_name_skipped(self):
        rows = gpu_usage._models_from_api([{"size": 100, "size_vram": 100},
                                           {"name": "ok", "size_vram": 100}])
        self.assertEqual([r["name"] for r in rows], ["ok"])


# ─── `ollama ps` table fallback ────────────────────────────────────────────

class OllamaPsTextTests(_GpuUsageBase):
    def test_parse_table_rows(self):
        rows = gpu_usage._parse_ollama_ps(_OLLAMA_PS_TEXT)
        self.assertEqual([r["name"] for r in rows],
                         ["qwen2.5:14b-instruct-q5_K_M", "qwen2.5vl:7b",
                          "nomic-embed-text:latest"])
        self.assertEqual(rows[0]["vram_mb"], 13 * 1024)         # 13 GB
        self.assertEqual(rows[1]["vram_mb"], int(7.3 * 1024))   # 7.3 GB
        self.assertEqual(rows[2]["vram_mb"], 323)               # 323 MB
        self.assertTrue(all(r["processor"] == "100% GPU" for r in rows))

    def test_parse_header_only_is_empty(self):
        header = _OLLAMA_PS_TEXT.splitlines()[0] + "\n"
        self.assertEqual(gpu_usage._parse_ollama_ps(header), [])

    def test_parse_cpu_gpu_split_processor(self):
        rows = gpu_usage._parse_ollama_ps(_OLLAMA_PS_SPILL)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["vram_mb"], 21 * 1024)
        self.assertEqual(rows[0]["processor"], "12%/88% CPU/GPU")

    # These three patch ``_run_ollama_ps``, NOT ``subprocess.run``.  They are
    # about the /api/ps -> `ollama ps` fallback LOGIC, so binding them to the
    # spawn mechanism was the wrong seam: when _run_ollama_ps moved off
    # subprocess.run (to make it actually bounded on Windows), the
    # subprocess.run mocks stopped intercepting anything and two of these went
    # green VACUOUSLY -- the real `ollama ps` ran and happened to return an
    # empty table, so "both sources down" passed while neither was down.
    def test_resident_models_prefers_api_over_text(self):
        # When /api/ps answers, the `ollama ps` fallback is never reached.
        ps = mock.MagicMock()
        with mock.patch.object(gpu_usage.urllib.request, "urlopen",
                               _fake_urlopen(_API_PS_BODY)), \
             mock.patch.object(gpu_usage, "_run_ollama_ps", ps):
            rows = gpu_usage._resident_models()
        self.assertEqual(len(rows), 3)
        ps.assert_not_called()

    def test_resident_models_falls_back_to_text(self):
        with mock.patch.object(gpu_usage.urllib.request, "urlopen",
                               side_effect=OSError("down")), \
             mock.patch.object(gpu_usage, "_run_ollama_ps",
                               return_value=_OLLAMA_PS_TEXT):
            rows = gpu_usage._resident_models()
        self.assertEqual([r["name"] for r in rows][:1],
                         ["qwen2.5:14b-instruct-q5_K_M"])

    def test_resident_models_empty_when_both_down(self):
        with mock.patch.object(gpu_usage.urllib.request, "urlopen",
                               side_effect=OSError("down")), \
             mock.patch.object(gpu_usage, "_run_ollama_ps",
                               return_value=None):
            self.assertEqual(gpu_usage._resident_models(), [])


class OllamaPsBoundednessTests(_GpuUsageBase):
    """`ollama ps` must be spawned by a construction that is bounded on Windows.

    Regression 2026-09-26 — the body used to be::

        subprocess.run(["ollama", "ps"], capture_output=True, timeout=_PS_TIMEOUT)

    which is NOT bounded on Windows.  When the timeout fires, CPython kills the
    child and then calls an **untimed** ``communicate()`` to drain the pipe (see
    the ``if _mswindows:`` branch of ``except TimeoutExpired`` in subprocess.py).
    The long-lived ollama SERVER holds the inherited write end of that pipe, so
    EOF never arrives and the drain never returns.

    Measured against a synthetic child that outlived the timeout and left a
    grandchild holding the pipe: the ``run()`` form never returned (>22 s), the
    file-redirect form returned in 2.01 s.  In the wild it wedged the entire
    test suite mid-run, reached from a "unit" test via build_status.
    """

    def _src(self) -> str:
        """The function's CODE, with the docstring stripped.

        The docstring deliberately names the forbidden construction so the next
        reader knows what not to go back to -- which tripped these assertions
        against the prose rather than the code.  Strip it via ast so the guard
        judges what actually executes.
        """
        import ast
        import inspect
        import textwrap
        fn = ast.parse(
            textwrap.dedent(inspect.getsource(gpu_usage._run_ollama_ps))).body[0]
        body = getattr(fn, "body", [])
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            fn.body = body[1:]
        return ast.unparse(fn)

    def test_does_not_use_the_unbounded_run_form(self):
        src = self._src()
        self.assertNotIn(
            "capture_output", src,
            "capture_output pipes stdout; on Windows the post-timeout drain of "
            "that pipe is UNTIMED and hangs when a grandchild holds it")
        self.assertNotIn(
            "subprocess.run", src,
            "subprocess.run's timeout is not a bound on Windows -- spawn with "
            "Popen and redirect stdout to a file instead")

    def test_redirects_stdout_to_a_file_and_waits_with_a_timeout(self):
        src = self._src()
        self.assertIn("tempfile.", src,
                      "stdout must go to a temp FILE so there is no pipe to "
                      "inherit and no reader thread to join")
        self.assertRegex(src, r"wait\(timeout=",
                         "the wait must carry the timeout")

    def test_timeout_kills_the_child_and_returns_none(self):
        """The timeout branch must kill the child and give up -- not drain."""
        exc = gpu_usage.subprocess.TimeoutExpired(cmd=["ollama", "ps"],
                                                 timeout=gpu_usage._PS_TIMEOUT)

        class _FakeProc:
            def __init__(self) -> None:
                self.killed = False

            def wait(self, timeout=None):
                if not self.killed:
                    raise exc
                return -9

            def kill(self) -> None:
                self.killed = True

        fake = _FakeProc()
        with mock.patch.object(gpu_usage.subprocess, "Popen",
                               return_value=fake):
            self.assertIsNone(gpu_usage._run_ollama_ps())
        self.assertTrue(fake.killed,
                        "a timed-out `ollama ps` child was never killed")


# ─── nvidia-smi CSV parsing ────────────────────────────────────────────────

class NvidiaSmiTests(_GpuUsageBase):
    def test_parse_single_gpu(self):
        out = gpu_usage._parse_nvidia_smi(_SMI_LINE)
        self.assertEqual(out["used_mb"], 23722)
        self.assertEqual(out["free_mb"], 605)
        self.assertEqual(out["total_mb"], 24576)
        self.assertEqual(out["util_pct"], 39)
        self.assertEqual(out["temp_c"], 33)

    def test_parse_multi_gpu_sums_mem_max_util_temp(self):
        out = gpu_usage._parse_nvidia_smi(_SMI_MULTI)
        self.assertEqual(out["used_mb"], 18000)
        self.assertEqual(out["total_mb"], 24000)
        self.assertEqual(out["util_pct"], 55)   # max of 20, 55
        self.assertEqual(out["temp_c"], 61)      # max of 40, 61

    def test_parse_garbage_yields_empty(self):
        self.assertEqual(gpu_usage._parse_nvidia_smi("N/A, N/A, N/A\n"), {})

    def test_run_query_none_on_missing_binary(self):
        with mock.patch.object(gpu_usage.subprocess, "run",
                               side_effect=FileNotFoundError()):
            self.assertIsNone(gpu_usage._run_nvidia_smi_query())

    def test_run_query_none_on_nonzero(self):
        with mock.patch.object(gpu_usage.subprocess, "run",
                               return_value=mock.MagicMock(returncode=9, stdout="")):
            self.assertIsNone(gpu_usage._run_nvidia_smi_query())


# ─── gpu_snapshot integration (all sources mocked) ─────────────────────────

class SnapshotTests(_GpuUsageBase):
    def _run_with(self, *, api=_API_PS_BODY, smi=_SMI_LINE,
                  urlopen_exc=None, smi_exc=None):
        """Run gpu_snapshot with /api/ps + nvidia-smi + `ollama ps` all mocked.

        nvidia-smi is routed by argv through subprocess.run; `ollama ps` is
        mocked at ``_run_ollama_ps`` instead.  It must NOT be routed through the
        argv mock: that helper deliberately does not use subprocess.run (it
        redirects stdout to a temp file so its timeout is bounded on Windows),
        so an argv router left the REAL binary reachable from a test that
        believed every source was mocked -- which is how a "no model source at
        all" case silently became "whatever ollama happens to say".
        """
        def _run(cmd, *a, **kw):
            exe = cmd[0] if cmd else ""
            if smi_exc is not None and exe == "nvidia-smi":
                raise smi_exc
            if exe == "nvidia-smi":
                return _smi_ok(smi) if smi is not None else mock.MagicMock(
                    returncode=1, stdout="")
            return mock.MagicMock(returncode=1, stdout="")

        # When /api/ps is forced down, make the `ollama ps` fallback
        # unavailable too, so the snapshot is exercised with NO model source.
        ps_text = None if urlopen_exc is not None else _OLLAMA_PS_TEXT

        url = (_fake_urlopen(OSError("x")) if urlopen_exc
               else _fake_urlopen(api))
        cm_url = (mock.patch.object(gpu_usage.urllib.request, "urlopen",
                                    side_effect=urlopen_exc) if urlopen_exc
                  else mock.patch.object(gpu_usage.urllib.request, "urlopen", url))
        with cm_url, mock.patch.object(gpu_usage.subprocess, "run", _run), \
             mock.patch.object(gpu_usage, "_run_ollama_ps",
                               return_value=ps_text):
            return gpu_usage.gpu_snapshot(use_cache=False)

    def test_full_snapshot(self):
        snap = self._run_with()
        self.assertEqual(snap["total_mb"], 24576)
        self.assertEqual(snap["used_mb"], 23722)
        self.assertEqual(snap["util_pct"], 39)
        self.assertEqual(snap["temp_c"], 33)
        self.assertEqual(len(snap["models"]), 3)
        self.assertEqual(snap["models"][0]["vram_mb"], 12832)
        self.assertIn("chat", snap["routing"])
        self.assertIsInstance(snap["ts"], float)

    def test_snapshot_no_nvidia_smi_is_partial(self):
        snap = self._run_with(smi_exc=FileNotFoundError())
        self.assertNotIn("total_mb", snap)
        # Models still come through /api/ps.
        self.assertEqual(len(snap["models"]), 3)
        self.assertIn("routing", snap)

    def test_snapshot_no_ollama_still_has_totals(self):
        snap = self._run_with(urlopen_exc=OSError("down"), smi=_SMI_LINE)
        # /api/ps down AND `ollama ps` returns nonzero here (routed to exit 1),
        # so no models — but nvidia-smi totals are present.
        self.assertEqual(snap["total_mb"], 24576)

    def test_snapshot_everything_missing_never_raises(self):
        snap = self._run_with(urlopen_exc=OSError("down"),
                              smi_exc=FileNotFoundError())
        self.assertEqual(snap["models"], [])
        self.assertNotIn("total_mb", snap)
        self.assertIn("routing", snap)   # routing always present

    def test_snapshot_uses_cache(self):
        calls = {"n": 0}

        def _run(cmd, *a, **kw):
            calls["n"] += 1
            return _smi_ok()

        with mock.patch.object(gpu_usage.urllib.request, "urlopen",
                               _fake_urlopen(_API_PS_BODY)), \
             mock.patch.object(gpu_usage.subprocess, "run", _run), \
             mock.patch.object(gpu_usage, "_run_ollama_ps", return_value=None):
            gpu_usage.gpu_snapshot(use_cache=False)   # primes cache
            n_after_first = calls["n"]
            gpu_usage.gpu_snapshot(use_cache=True)     # served from cache
        self.assertEqual(calls["n"], n_after_first)    # no extra subprocess


# ─── formatting: usage_lines / usage_bar / routing_line ────────────────────

class FormatTests(_GpuUsageBase):
    def _snap(self, **over):
        snap = {
            "total_mb": 24576, "used_mb": 20600, "free_mb": 3976,
            "util_pct": 29, "temp_c": 41,
            "models": [
                {"name": "qwen2.5:14b", "vram_mb": 13312, "size_mb": 13312,
                 "processor": "100% GPU"},
                {"name": "qwen2.5vl:7b", "vram_mb": 7475, "size_mb": 7475,
                 "processor": "100% GPU"},
            ],
            "routing": {"chat": "cloud", "vision": "cloud", "ambient": "local"},
            "ts": 1.0,
        }
        snap.update(over)
        return snap

    def test_usage_lines_per_model_rows(self):
        lines = gpu_usage.usage_lines(self._snap())
        self.assertEqual(lines[0], "qwen2.5:14b  13.0/24 GB")
        self.assertEqual(lines[1], "qwen2.5vl:7b  7.3/24 GB")

    def test_usage_lines_total_line(self):
        lines = gpu_usage.usage_lines(self._snap())
        total = [ln for ln in lines if ln.startswith("TOTAL")][0]
        self.assertIn("20.1/24 GB", total)     # 20600 MB ≈ 20.1 GB
        self.assertIn("(84%)", total)          # 20600/24576 ≈ 84%
        self.assertIn("util 29%", total)
        self.assertIn("41C", total)

    def test_usage_lines_routing_line(self):
        lines = gpu_usage.usage_lines(self._snap())
        self.assertEqual(lines[-1], "chat→cloud  vision→cloud  ambient→local")

    def test_usage_lines_cpu_spill_annotated(self):
        snap = self._snap(models=[
            {"name": "qwen2.5:32b", "vram_mb": 18000, "size_mb": 21000,
             "processor": "14%/86% CPU/GPU"},
        ])
        lines = gpu_usage.usage_lines(snap)
        self.assertIn("[14%/86% CPU/GPU]", lines[0])

    def test_usage_lines_no_models(self):
        snap = self._snap(models=[])
        lines = gpu_usage.usage_lines(snap)
        self.assertIn("no models resident on the GPU", lines[0])
        # TOTAL still present from nvidia-smi used_mb.
        self.assertTrue(any(ln.startswith("TOTAL") for ln in lines))

    def test_usage_lines_no_nvidia_smi(self):
        # No total/used → per-model rows show just "<gb> GB", TOTAL falls back
        # to the summed model VRAM (no percent).
        snap = {"models": [{"name": "m", "vram_mb": 4096, "size_mb": 4096,
                            "processor": "100% GPU"}],
                "routing": {"chat": "auto", "vision": "auto", "ambient": "auto"},
                "ts": 1.0}
        lines = gpu_usage.usage_lines(snap)
        self.assertEqual(lines[0], "m  4.0 GB")
        total = [ln for ln in lines if ln.startswith("TOTAL")][0]
        self.assertIn("4.0 GB", total)
        self.assertNotIn("%", total)

    def test_routing_line_default_auto(self):
        snap = {"routing": {"chat": "auto", "vision": "auto", "ambient": "auto"}}
        self.assertEqual(gpu_usage.routing_line(snap),
                         "chat→auto  vision→auto  ambient→auto")

    def test_usage_bar_partial(self):
        # 20600 / 24576 ≈ 84% → 17/20 cells.
        bar = gpu_usage.usage_bar(20, self._snap())
        self.assertTrue(bar.startswith("[") and bar.endswith("84%"))
        self.assertEqual(bar.count("#"), 17)
        self.assertEqual(bar.count("-"), 3)

    def test_usage_bar_na_without_total(self):
        snap = {"models": [], "routing": {}, "ts": 1.0}
        self.assertIn("n/a", gpu_usage.usage_bar(20, snap))

    def test_usage_bar_full(self):
        snap = self._snap(used_mb=24576)
        bar = gpu_usage.usage_bar(10, snap)
        self.assertEqual(bar, "[##########] 100%")

    def test_usage_summary_text_joins_lines(self):
        text = gpu_usage.usage_summary_text(self._snap())
        self.assertIn("  ·  ", text)
        self.assertIn("TOTAL", text)
        self.assertIn("chat→cloud", text)


# ─── numeric helpers ────────────────────────────────────────────────────────

class HelperTests(_GpuUsageBase):
    def test_bytes_to_mb(self):
        self.assertEqual(gpu_usage._bytes_to_mb(1024 * 1024), 1)
        self.assertEqual(gpu_usage._bytes_to_mb(None), 0)

    def test_size_pair_to_mb_units(self):
        self.assertEqual(gpu_usage._size_pair_to_mb("13", "GB"), 13 * 1024)
        self.assertEqual(gpu_usage._size_pair_to_mb("323", "MB"), 323)
        self.assertEqual(gpu_usage._size_pair_to_mb("2", "TB"), 2 * 1024 * 1024)
        self.assertEqual(gpu_usage._size_pair_to_mb("x", "GB"), 0)
        self.assertEqual(gpu_usage._size_pair_to_mb("5", "??"), 0)

    def test_looks_numeric(self):
        self.assertTrue(gpu_usage._looks_numeric("7.3"))
        self.assertTrue(gpu_usage._looks_numeric("13"))
        self.assertFalse(gpu_usage._looks_numeric("GB"))
        self.assertFalse(gpu_usage._looks_numeric("1.2.3"))


if __name__ == "__main__":
    unittest.main()
