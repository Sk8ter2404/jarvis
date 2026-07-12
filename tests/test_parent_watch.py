"""core/parent_watch.py — definitive parent liveness for overlay children.

Born 2026-07-12: four overlay processes (unified HUD, tray, reticle,
air-cursor) each trusted psutil.pid_exists, which reads TRUE for BOTH
Windows dead states — terminated-but-unreaped (a held handle keeps the row)
and terminating-forever (thread pinned in a kernel driver after the
ExitProcess loader-lock hang). The owner saw two of everything for 25
minutes. The unreaped case is deterministic to reproduce: a Popen child that
has exited while our Popen object still holds its handle.
"""
import os
import subprocess
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.parent_watch import parent_is_alive  # noqa: E402


class ParentIsAliveTests(unittest.TestCase):
    def test_no_parent_convention(self):
        # pid <= 0 = "nothing to watch" — every overlay treats that as alive.
        self.assertTrue(parent_is_alive(0))
        self.assertTrue(parent_is_alive(-1))
        self.assertTrue(parent_is_alive(None))

    def test_own_process_is_alive(self):
        self.assertTrue(parent_is_alive(os.getpid()))

    def test_nonexistent_pid_is_dead(self):
        # Beyond any plausible live pid on this box.
        self.assertFalse(parent_is_alive(0x7FFFFFF0))

    def test_exited_child_reads_dead_while_handle_held(self):
        # THE regression: our Popen object holds a handle, so the child's
        # process row stays enumerable after exit — psutil.pid_exists reads
        # True (that's the bug that stranded the overlays), parent_is_alive
        # must read False.
        child = subprocess.Popen(
            [sys.executable, "-c", "pass"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            child.wait(timeout=30)          # exited; handle still open
            self.assertFalse(
                parent_is_alive(child.pid),
                "an exited-but-unreaped child must read DEAD")
            if sys.platform == "win32":
                try:
                    import psutil
                    # Document the psutil behaviour this module exists to
                    # correct (informational — don't fail if psutil evolves).
                    if psutil.pid_exists(child.pid):
                        pass  # the exact false-positive the helper fixes
                except ImportError:
                    pass
        finally:
            child.poll()


if __name__ == "__main__":
    unittest.main()
