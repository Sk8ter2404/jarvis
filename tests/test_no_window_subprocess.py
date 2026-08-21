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
from unittest import mock


from core import no_window_subprocess as nw

CNW = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)


@unittest.skipUnless(os.name == "nt", "Windows-only behaviour")
class NoWindowNetTests(unittest.TestCase):
    def setUp(self):
        # Replace the STOCK Popen.__init__ with a recorder BEFORE install()
        # wraps it (the net closes over whatever __init__ it finds, so the
        # recorder must already be in place) — no process is ever spawned.
        #
        # The recorder goes on the BASE class, which is where install() now
        # puts the net. Writing it onto whatever `subprocess.Popen` NAMES would
        # leave an unmarked shadow on tools/browser_guard.py's GuardedPopen for
        # the rest of the run — i.e. this test file would disarm the very net it
        # is testing, for every module that runs after it.
        nw.uninstall()
        base = nw._popen_base()
        self.assertIsNotNone(base, "subprocess.Popen is not a class")
        self.seen: dict = {}
        rec_self = self

        def _record(popen_self, *a, **k):
            rec_self.seen = k

        saved_own = base.__dict__.get("__init__")
        # LIFO cleanups: uninstall() runs FIRST (drops our wrappers), then the
        # class's own __init__ is put back exactly as it was, then the net is
        # re-armed so the REST of the run stays protected.
        self.addCleanup(nw.install)
        self.addCleanup(lambda: setattr(base, "__init__", saved_own))
        self.addCleanup(nw.uninstall)
        base.__init__ = _record
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


@unittest.skipUnless(os.name == "nt", "Windows-only behaviour")
class RebindSurvivalTests(unittest.TestCase):
    """THE DEFECT (2026-08-20, adversarial review of the same day's guard work)

    ``install()`` wrote ``_no_window_init`` onto whatever class
    ``subprocess.Popen`` NAMED at that moment. During a test run that is
    ``tools/browser_guard.py``'s ``GuardedPopen`` subclass — and
    ``browser_guard._reset_for_tests()`` restores the STOCK class before
    ``install()`` builds a brand-new subclass, so the net went with the
    discarded one. ``install()`` then short-circuited on ``_ORIG_INIT[0] is not
    None`` and returned True — "already installed" — while nothing was
    installed.

    Consequence for every full local run: from ``tests.test_browser_guard``
    onward, any production code under test that calls a bare
    ``subprocess.run(...)`` with no creationflags spawns with a console
    attached — the '~38 ghost Windows Terminal windows in 30 minutes' incident
    this module exists to prevent, on the owner's desktop, with the module
    reporting itself active.

    The fix is the one the live-data guard already learned: patch the class at
    the BOTTOM of the MRO (no rebind can replace it), mark the wrapper, and make
    ``install()`` repair-only rather than latch on a stored original."""

    def setUp(self):
        self.seen: dict = {}
        rec = self

        def _record(popen_self, *a, **k):
            rec.seen = k

        self._record = _record
        base = nw._popen_base()
        self.assertIsNotNone(base)
        self.base = base
        self._saved_base_init = base.__dict__.get("__init__")
        self._saved_popen = subprocess.Popen
        self.addCleanup(self._restore)
        nw.uninstall()
        base.__init__ = _record
        nw.install()

    def _restore(self):
        nw.uninstall()
        subprocess.Popen = self._saved_popen
        if self._saved_base_init is not None:
            self.base.__init__ = self._saved_base_init
        nw.install()

    def test_the_net_survives_a_popen_rebind_to_a_fresh_subclass(self):
        """The literal browser_guard cycle: swap in a stock-derived subclass,
        the way ``_reset_for_tests()`` + ``install()`` does."""
        class GuardedPopen(self.base):
            pass

        subprocess.Popen = GuardedPopen
        self.assertTrue(nw.is_armed(),
                        "rebinding subprocess.Popen threw the no-window net "
                        "away, and install() still reports it installed")
        subprocess.Popen(["whatever.exe"])
        self.assertEqual(self.seen.get("creationflags"), CNW)

    def test_is_armed_is_stricter_than_install_returning_true(self):
        """``install()`` returning True must never be the ONLY evidence: that
        is the exact lie this net told for a whole CI run."""
        saved = self.base.__dict__.get("__init__")
        try:
            self.base.__init__ = self._record          # displaced
            self.assertFalse(nw.is_armed())
            self.assertTrue(nw.install())              # ...and repaired
            self.assertTrue(nw.is_armed())
            subprocess.Popen(["whatever.exe"])
            self.assertEqual(self.seen.get("creationflags"), CNW)
        finally:
            if saved is not None:
                self.base.__init__ = saved

    def test_install_repairs_a_shadowing_subclass_init(self):
        """A subclass that defines its OWN ``__init__`` shadows the base patch
        for attribute lookup. install() has to repair that too, or it reports
        armed while every spawn bypasses the net."""
        class Shadowed(self.base):
            pass

        stock = self._record
        Shadowed.__init__ = lambda self, *a, **k: stock(self, *a, **k)
        subprocess.Popen = Shadowed
        self.assertFalse(nw.is_armed())
        nw.install()
        self.assertTrue(nw.is_armed())
        subprocess.Popen(["whatever.exe"])
        self.assertEqual(self.seen.get("creationflags"), CNW)

    def test_install_never_double_wraps(self):
        """Repair-only: a second call must not stack a second wrapper (which
        would still 'work' but grows without bound across entry points)."""
        before = self.base.__dict__.get("__init__")
        nw.install()
        nw.install()
        self.assertIs(self.base.__dict__.get("__init__"), before)

    def test_is_armed_is_not_fooled_by_a_magicmock(self):
        saved = self.base.__dict__.get("__init__")
        try:
            self.base.__init__ = mock.MagicMock()
            self.assertFalse(nw.is_armed())
        finally:
            if saved is not None:
                self.base.__init__ = saved


@unittest.skipUnless(os.name == "nt", "Windows-only behaviour")
class ChainMarkerTests(unittest.TestCase):
    def test_marked_walks_the_wrapper_chain(self):
        """A SIBLING guard wrapping our net (tests/live_data_guard.py wraps
        ``Popen.__init__`` too) must not read as a displacement — otherwise two
        correctly-cooperating guards report as one broken one, and the repair
        path re-wraps forever."""
        def ours(self, *a, **k):
            return None
        setattr(ours, nw._GUARD_MARK, True)

        def sibling(self, *a, **k):
            return ours(self, *a, **k)
        sibling.__wrapped__ = ours

        self.assertTrue(nw._marked(ours))
        self.assertTrue(nw._marked(sibling))
        self.assertFalse(nw._marked(lambda self, *a, **k: None))
        self.assertFalse(nw._marked(mock.MagicMock()))



if __name__ == "__main__":
    unittest.main()
