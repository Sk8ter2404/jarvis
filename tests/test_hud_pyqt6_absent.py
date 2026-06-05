"""Every hud/ module MUST import cleanly even when PyQt6 is absent.

The launcher runs each HUD as a subprocess; a module that subclasses a Qt base
class (``class X(QWidget)``) or builds a ``QColor(...)`` at module scope without
stubbing the Qt names in its ``except ImportError`` fallback raises NameError at
IMPORT time -- before its ``if not _HAS_PYQT6: ... sys.exit(2)`` guard in main()
ever runs. The launcher then sees a crash instead of the intended clean
"install PyQt6" exit. This test forces PyQt6 absent and imports every hud module
to guard against that regression (the self-upgrade pipeline rewrites these files
often, so the guard must be enforced by a test).
"""
import importlib
import pathlib
import sys
import unittest

_HUD_DIR = pathlib.Path(__file__).resolve().parent.parent / "hud"
_HUD_MODULES = sorted(
    "hud." + f.stem for f in _HUD_DIR.glob("*.py") if not f.stem.startswith("__")
)
_QT = ("PyQt6", "PyQt6.QtCore", "PyQt6.QtGui", "PyQt6.QtWidgets", "PyQt6.QtSvg")


class HudImportsWithoutPyQt6(unittest.TestCase):
    def test_all_hud_modules_import_without_pyqt6(self):
        self.assertTrue(_HUD_MODULES, "no hud modules were discovered")
        saved_qt = {k: sys.modules.get(k) for k in _QT}
        saved_hud = {m: sys.modules.pop(m, None) for m in _HUD_MODULES}
        for k in _QT:
            sys.modules[k] = None          # force ImportError on `from PyQt6... import`
        try:
            for m in _HUD_MODULES:
                with self.subTest(module=m):
                    try:
                        mod = importlib.import_module(m)
                    except BaseException as e:  # any import failure is the bug
                        self.fail(f"{m} failed to import without PyQt6: "
                                  f"{type(e).__name__}: {e}")
                    self.assertFalse(
                        getattr(mod, "_HAS_PYQT6", False),
                        f"{m} should report _HAS_PYQT6=False when PyQt6 is absent",
                    )
        finally:
            for m in _HUD_MODULES:
                sys.modules.pop(m, None)    # drop the PyQt6-less copies
            for k in _QT:
                if sys.modules.get(k) is None:
                    sys.modules.pop(k, None)
            for k, v in saved_qt.items():
                if v is not None:
                    sys.modules[k] = v
            for m, v in saved_hud.items():
                if v is not None:
                    sys.modules[m] = v


if __name__ == "__main__":
    unittest.main()
