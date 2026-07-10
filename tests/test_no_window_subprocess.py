"""core/no_window_subprocess — the process-wide CREATE_NO_WINDOW safety net.

The ghost-window class (pythonw spawning console apps with no flag → visible
Windows Terminal windows piling up) returned within hours of the per-site
v2.0.32 fixes via unaudited spawn sites, so the net patches Popen once.
These tests verify the flag-defaulting logic WITHOUT launching processes —
the patched __init__ is intercepted before the real one runs."""
from __future__ import annotations

import os
import subprocess
import unittest


from core import no_window_subprocess as nw

CNW = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)


@unittest.skipUnless(os.name == "nt", "Windows-only behaviour")
class NoWindowNetTests(unittest.TestCase):
    def setUp(self):
        # Replace the STOCK Popen.__init__ with a recorder BEFORE install()
        # wraps it (the net closes over whatever __init__ it finds, so the
        # recorder must already be in place) — no process is ever spawned.
        nw.uninstall()
        self.seen: dict = {}
        rec_self = self

        def _record(popen_self, *a, **k):
            rec_self.seen = k

        real_init = subprocess.Popen.__init__
        # LIFO cleanups: uninstall() runs FIRST (unwraps back to the
        # recorder), then the real __init__ is restored.
        self.addCleanup(lambda: setattr(subprocess.Popen, "__init__", real_init))
        self.addCleanup(nw.uninstall)
        subprocess.Popen.__init__ = _record
        nw.install()

    def test_bare_spawn_gets_no_window(self):
        subprocess.Popen(["whatever.exe"])
        self.assertEqual(self.seen.get("creationflags"), CNW)

    def test_explicit_flags_pass_untouched(self):
        detached = getattr(subprocess, "DETACHED_PROCESS", 0x8)
        subprocess.Popen(["whatever.exe"], creationflags=detached)
        self.assertEqual(self.seen.get("creationflags"), detached)

    def test_startupinfo_respected(self):
        si = subprocess.STARTUPINFO()
        subprocess.Popen(["whatever.exe"], startupinfo=si)
        self.assertIs(self.seen.get("startupinfo"), si)
        self.assertNotIn("creationflags", self.seen)

    def test_run_routes_through_net(self):
        # subprocess.run builds a Popen internally — same net applies. run()
        # will fail after __init__ (recorder returns None attrs); we only
        # care that the flag was injected before that.
        try:
            subprocess.run(["whatever.exe"], timeout=0.1)
        except Exception:
            pass
        self.assertEqual(self.seen.get("creationflags"), CNW)

    def test_install_is_idempotent(self):
        first = nw._ORIG_INIT[0]
        self.assertTrue(nw.install())     # second install: no re-wrap
        self.assertIs(nw._ORIG_INIT[0], first)


if __name__ == "__main__":
    unittest.main()
