"""Structural safety net: every Python file in the project must byte-compile,
and the import-light core modules must import cleanly. This is the fast
regression catch the self-upgrade pipeline cares about most — a syntax error or
load-time crash in any skill/core/tool file fails here in seconds, without a
full boot."""
import importlib
import os
import py_compile
import unittest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Directories that hold shippable source. Everything else (backups, caches,
# venvs, the staging sandbox) is excluded from the compile sweep.
_SOURCE_DIRS = ("", "core", "skills", "tools", "hud", "adapters", "tts", "audio")
_EXCLUDE_DIR_NAMES = {
    "backups", "__pycache__", ".git", "data_staging", "venv", ".venv",
    "node_modules", "data", "logs", ".pytest_cache",
}

# Core modules that must always import without booting the monolith. Keep this
# list to the genuinely import-light ones (no Whisper/CUDA/audio at import).
_IMPORT_LIGHT_CORE = (
    "core.config", "core.state", "core.atomic_io", "core.prompts",
    "core.mode_router", "core.long_term_memory", "core.memory",
    # Modules extracted from the monolith — must import cleanly (stdlib-only or
    # core-only) so a wrong name surfaces here, not at JARVIS boot.
    "core.tts", "core.llm_client", "core.tone_detector",
    "core.speech_filter", "core.voice_emotion", "core.memory_guards",
    "core.legacy_memory", "core.stream_speech",
)


def _iter_source_files():
    for rel in _SOURCE_DIRS:
        base = os.path.join(_PROJECT_ROOT, rel) if rel else _PROJECT_ROOT
        if not os.path.isdir(base):
            continue
        if rel == "":
            # Root: only top-level .py files (don't recurse into sibling dirs
            # here — each source dir is walked on its own pass).
            for name in os.listdir(base):
                p = os.path.join(base, name)
                if name.endswith(".py") and os.path.isfile(p):
                    yield p
            continue
        for root, dirs, files in os.walk(base):
            dirs[:] = [d for d in dirs if d not in _EXCLUDE_DIR_NAMES]
            for name in files:
                if name.endswith(".py"):
                    yield os.path.join(root, name)


class CompileSweepTests(unittest.TestCase):
    def test_all_sources_compile(self):
        failures = []
        count = 0
        for path in _iter_source_files():
            count += 1
            try:
                py_compile.compile(path, doraise=True)
            except py_compile.PyCompileError as exc:
                failures.append(f"{os.path.relpath(path, _PROJECT_ROOT)}: {exc.msg}")
        self.assertGreater(count, 50, "expected to sweep >50 source files")
        self.assertEqual(failures, [], "files failed to compile:\n" + "\n".join(failures))


class CoreImportTests(unittest.TestCase):
    def test_import_light_core_modules_load(self):
        failures = []
        for mod in _IMPORT_LIGHT_CORE:
            try:
                importlib.import_module(mod)
            except Exception as exc:  # noqa: BLE001 — we want every failure listed
                failures.append(f"{mod}: {type(exc).__name__}: {exc}")
        self.assertEqual(failures, [], "core modules failed to import:\n" + "\n".join(failures))


if __name__ == "__main__":
    unittest.main()
