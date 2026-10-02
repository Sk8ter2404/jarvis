"""End to end: tools/action_smoke.py sweeps inside a throwaway copy (2026-10-01).

The 09-05 live diagnostic: running the sweep made the LIVE JARVIS speak fake
alerts — its monolith wrote the pending-speech queue the live loop drains.
This drives the real tool as a subprocess (the sandbox copy, the forced
redirects, the child's preflight probe, a two-action sweep) and checks the
tree it was run from is untouched. The pure pieces are tests/test_action_smoke.py.

    python -m unittest tests.monolith.test_monolith_action_smoke_sandbox
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest

from tests import live_data_guard
from tests._monolith_harness import requires_monolith

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Named through the guard (tests may not build a data/ path off the root).
_LIVE_DATA = live_data_guard.LIVE_DATA_DIR
_TOOL = os.path.join(_ROOT, "tools", "action_smoke.py")
# Root files a sweep must never create or change in the tree it runs from.
_WATCHED = ("pending_speech.json", "injected_commands.json",
            "tray_commands.json", "jarvis_todo.md", "hud_state.json")


def _stamp(path):
    try:
        st = os.stat(path)
        return (st.st_mtime_ns, st.st_size)
    except OSError:
        return None


@requires_monolith
class ActionSmokeSandboxEndToEndTests(unittest.TestCase):
    def _run(self, *args, timeout=420):
        before = {n: _stamp(os.path.join(_ROOT, n)) for n in _WATCHED}
        before_data = _stamp(_LIVE_DATA)
        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        proc = subprocess.run([sys.executable, "-B", _TOOL, *args],
                              cwd=_ROOT, env=env, capture_output=True,
                              text=True, encoding="utf-8", errors="replace",
                              timeout=timeout)
        after = {n: _stamp(os.path.join(_ROOT, n)) for n in _WATCHED}
        self.assertEqual(before, after,
                         "the sweep created or changed live root state")
        self.assertEqual(before_data, _stamp(_LIVE_DATA))
        return proc

    def test_preflight_proves_the_queue_is_the_sandbox_queue(self):
        proc = self._run("--preflight-only")
        out = proc.stdout + proc.stderr
        self.assertEqual(proc.returncode, 0, out[-3000:])
        self.assertIn("preflight OK", out)
        self.assertIn("[smoke] sandbox:", out)
        # The child's live-data guard protects THIS tree, not its own copy.
        self.assertIn(f"armed on {_LIVE_DATA}", out)

    def test_a_small_sweep_runs_and_cleans_up(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = os.path.join(tmp, "smoke.json")
            proc = self._run("--no-skills", "--only", "get_time,version_info",
                             "--json", results)
            out = proc.stdout + proc.stderr
            self.assertEqual(proc.returncode, 0, out[-3000:])
            with open(results, encoding="utf-8") as f:
                got = json.load(f)
        self.assertEqual(sorted(got["ok"]), ["get_time", "version_info"])
        self.assertEqual(got["blocked_real_tree_writes"], [])
        sandbox = [ln.split("sandbox:", 1)[1].strip()
                   for ln in out.splitlines() if "[smoke] sandbox:" in ln]
        self.assertTrue(sandbox)
        self.assertFalse(os.path.exists(sandbox[0]), "the copy was left behind")


if __name__ == "__main__":
    unittest.main()
