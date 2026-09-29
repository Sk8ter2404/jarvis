"""bobert_companion's import-time stdout/stderr UTF-8 conversion must run on the
MAIN THREAD ONLY, and never twice.

THE DEFECT THIS PINS (v2.0.128 CI, 2026-09-29). The monolith's first statements
``reconfigure()`` sys.stdout and sys.stderr. CPython's TextIOWrapper.reconfigure
is not atomic: it drops the stream's encoder, runs Python code to build the new
one, then installs it, so a ``write()`` from another thread in that gap raises
``io.UnsupportedOperation: not writable``. ~40 modules late-bind the monolith
with ``importlib.import_module("bobert_companion")``, several from daemon
threads, and wherever that import FAILS (the Linux CI runner has no
sounddevice/cv2) nothing is cached, so every attempt re-ran the conversion. A
game-mode poller leaked by tests/skills/test_game_mode_deferred_restore.py did
it ~800 times in one ci-sim run, and on CI one landed while the unittest runner
was writing a "." to sys.stderr — the coverage step died mid-suite with that
exact exception in TextTestResult.addSuccess.

Light tier by design: the monolith cannot import on the bare runner, so this
compiles the ONE top-level statement that calls ``.reconfigure(`` straight out
of bobert_companion.py and runs it against fake streams. It is the shipped code,
not a copy of it, so a revert of the guard fails here. stdlib unittest only.
"""
from __future__ import annotations

import ast
import codecs
import functools
import os
import threading
import types
import unittest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MONOLITH = os.path.join(_ROOT, "bobert_companion.py")


@functools.lru_cache(maxsize=1)
def _stdio_statements() -> tuple:
    """Every TOP-LEVEL statement of the monolith that calls ``.reconfigure``.
    Parsed once per process: the monolith is ~30K lines."""
    with open(_MONOLITH, encoding="utf-8") as f:
        tree = ast.parse(f.read(), _MONOLITH)
    return tuple(node for node in tree.body
                 if any(isinstance(n, ast.Attribute) and n.attr == "reconfigure"
                        for n in ast.walk(node)))


class _Stream:
    """A text stream that records reconfigure() calls instead of making them."""

    def __init__(self, encoding: str, errors: str):
        self.encoding = encoding
        self.errors = errors
        self.calls: list[dict] = []

    def reconfigure(self, **kw):
        self.calls.append(kw)
        self.encoding = kw.get("encoding", self.encoding)
        self.errors = kw.get("errors", self.errors)


def _run(stdout, stderr, *, on_worker_thread: bool = False) -> None:
    """Execute the monolith's stdio statement with ``sys`` swapped for a fake
    carrying these two streams. Re-raises anything it raised."""
    (node,) = _stdio_statements()
    code = compile(ast.Module(body=[node], type_ignores=[]), _MONOLITH, "exec")
    ns = {"sys": types.SimpleNamespace(stdout=stdout, stderr=stderr),
          "codecs": codecs, "threading": threading}
    raised: list[BaseException] = []

    def _body():
        try:
            exec(code, ns)
        except BaseException as e:  # noqa: BLE001 - surfaced to the test below
            raised.append(e)

    if on_worker_thread:
        th = threading.Thread(target=_body, name="lazy-importer")
        th.start()
        th.join(timeout=10)
    else:
        _body()
    if raised:
        raise raised[0]


class MonolithStdioConversionTests(unittest.TestCase):

    def test_exactly_one_top_level_statement_converts_stdio(self):
        stmts = _stdio_statements()
        self.assertEqual(len(stmts), 1,
                         "expected ONE import-time stdio conversion in "
                         "bobert_companion.py; found %d" % len(stmts))

    def test_main_thread_import_still_converts_a_legacy_console(self):
        """The reason the block exists: a cp1252 console must come out UTF-8
        with errors='replace', so an em-dash can never raise at boot."""
        out, err = _Stream("cp1252", "strict"), _Stream("cp1252", "backslashreplace")
        _run(out, err)
        for s in (out, err):
            self.assertEqual(s.calls, [{"encoding": "utf-8", "errors": "replace"}])

    def test_a_re_import_leaves_a_converted_stream_alone(self):
        """A failed import is never cached, so the block re-runs on every
        attempt; rebuilding a LIVE stream's encoder each time is the race."""
        for spelling in ("utf-8", "UTF-8", "utf8", "UTF_8"):
            with self.subTest(encoding=spelling):
                out, err = _Stream(spelling, "replace"), _Stream(spelling, "replace")
                _run(out, err)
                self.assertEqual((out.calls, err.calls), ([], []))

    def test_utf8_with_a_different_error_handler_is_still_converted(self):
        """Already UTF-8 but 'strict' can still raise on a lone surrogate, so
        the one conversion a main-thread boot owes is still made."""
        out, err = _Stream("utf-8", "strict"), _Stream("utf-8", "backslashreplace")
        _run(out, err)
        self.assertEqual(len(out.calls), 1)
        self.assertEqual(len(err.calls), 1)

    def test_an_import_from_a_worker_thread_never_touches_stdio(self):
        """THE v2.0.128 path: a daemon thread's lazy import_module must not
        rebuild the encoder of a stream the main thread is writing to."""
        out, err = _Stream("cp1252", "strict"), _Stream("cp1252", "backslashreplace")
        _run(out, err, on_worker_thread=True)
        self.assertEqual((out.calls, err.calls), ([], []))

    def test_streams_that_cannot_be_converted_are_tolerated(self):
        """None (pythonw), an object with no encoding and a stream whose
        reconfigure() raises must all be survived: this runs at import."""

        class _Raises(_Stream):
            def reconfigure(self, **kw):
                raise OSError("cannot reconfigure")

        _run(None, object())
        _run(_Raises("cp1252", "strict"), _Raises("cp1252", "strict"))


if __name__ == "__main__":
    unittest.main()
