"""Audit A86: a malformed numeric environment value must not stop JARVIS from
booting.

core/config.py parsed JARVIS_AUDIO_POLL_S and JARVIS_AUDIO_MIC_SILENT_S with a
bare ``float(os.getenv(...))`` at import time. The monolith imports core.config
at module top level, so ``JARVIS_AUDIO_POLL_S=abc`` raised ValueError on import
and JARVIS never started. Now a value that is not a finite number falls back to
the shipped default and prints ONE warning line naming the variable.

The reload pattern matches tests/test_config.py: tearDown reloads the module
under the ambient environment so nothing leaks into later tests. The settings
file is hidden during the reload so a saved override cannot mask the default.
stdlib unittest + importlib only.
"""
from __future__ import annotations

import contextlib
import importlib
import io
import math
import os
import unittest
from unittest import mock

from core import config

_REAL_EXISTS = os.path.exists


def _no_user_settings(path):
    if str(path).replace("\\", "/").endswith("data/user_settings.json"):
        return False
    return _REAL_EXISTS(path)


class MalformedEnvFloatTests(unittest.TestCase):
    def tearDown(self):
        importlib.reload(config)

    def _reload(self, **env):
        err = io.StringIO()
        with mock.patch.dict(os.environ, env, clear=False), \
             mock.patch("os.path.exists", _no_user_settings), \
             contextlib.redirect_stderr(err), \
             contextlib.redirect_stdout(io.StringIO()):
            mod = importlib.reload(config)
        return mod, err.getvalue()

    def test_bad_values_fall_back_to_the_defaults_instead_of_stopping_boot(self):
        cases = [
            ("JARVIS_AUDIO_POLL_S", "AUDIO_AUTOSWITCH_POLL_S", 3.0),
            ("JARVIS_AUDIO_MIC_SILENT_S", "AUDIO_AUTOSWITCH_MIC_SILENT_S", 60.0),
        ]
        for env_name, const, default in cases:
            for bad in ("abc", "3,5", "nan", "inf", "-inf", "1e999"):
                with self.subTest(var=env_name, value=bad):
                    mod, err = self._reload(**{env_name: bad})
                    value = getattr(mod, const)
                    self.assertIsInstance(value, float)
                    self.assertEqual(value, default)
                    lines = [ln for ln in err.splitlines() if env_name in ln]
                    self.assertEqual(len(lines), 1, err)

    def test_good_values_still_apply(self):
        mod, err = self._reload(JARVIS_AUDIO_POLL_S="2.5",
                                JARVIS_AUDIO_MIC_SILENT_S=" 0 ")
        self.assertEqual(mod.AUDIO_AUTOSWITCH_POLL_S, 2.5)
        self.assertEqual(mod.AUDIO_AUTOSWITCH_MIC_SILENT_S, 0.0)
        self.assertNotIn("JARVIS_AUDIO_POLL_S", err)
        self.assertNotIn("JARVIS_AUDIO_MIC_SILENT_S", err)

    def test_unset_uses_the_default_silently(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("JARVIS_AUDIO_POLL_S", None)
            os.environ.pop("JARVIS_AUDIO_MIC_SILENT_S", None)
            mod, err = self._reload()
        self.assertEqual(mod.AUDIO_AUTOSWITCH_POLL_S, 3.0)
        self.assertEqual(mod.AUDIO_AUTOSWITCH_MIC_SILENT_S, 60.0)
        self.assertTrue(math.isfinite(mod.AUDIO_AUTOSWITCH_POLL_S))
        self.assertNotIn("JARVIS_AUDIO_POLL_S", err)

    def test_blank_counts_as_unset(self):
        mod, err = self._reload(JARVIS_AUDIO_POLL_S="  ",
                                JARVIS_AUDIO_MIC_SILENT_S="")
        self.assertEqual(mod.AUDIO_AUTOSWITCH_POLL_S, 3.0)
        self.assertEqual(mod.AUDIO_AUTOSWITCH_MIC_SILENT_S, 60.0)
        self.assertNotIn("JARVIS_AUDIO_POLL_S", err)
        self.assertNotIn("JARVIS_AUDIO_MIC_SILENT_S", err)


if __name__ == "__main__":
    unittest.main()
