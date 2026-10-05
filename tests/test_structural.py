"""Structural safety net: every Python file in the project must byte-compile,
and the import-light core modules must import cleanly. This is the fast
regression catch the self-upgrade pipeline cares about most — a syntax error or
load-time crash in any skill/core/tool file fails here in seconds, without a
full boot."""
import importlib
import os
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
    "core.followup_window", "core.ollama_opts", "core.processing_filler",
    "core.owner_turn",
    "core.audio_flap",
    # The camera open gate (2026-09-29): stdlib-only, imported at monolith
    # import time before the Kinect bridge is enabled.
    "core.camera_gate",
    # The CUDA driver pre-init and the CTranslate2 host thread (v2.0.180, the
    # v2.0.179 thread-exit abort): stdlib-only, imported at monolith import
    # time - they must never pull in ctranslate2 or a CUDA library.
    "core.cuda_preinit", "core.ct2_host",
    # The capture-open backoff (R10, 2026-09-29): stdlib-only, imported at
    # monolith import time.
    "core.input_backoff",
    # The proactive-remark text / pacing gate and the sustained face-presence
    # tracker (2026-09-30): stdlib-only, imported at monolith import time.
    "core.proactive_guard", "core.face_presence",
    # The web dashboard's camera tile roster + skill panel registry
    # (2026-09-30): stdlib-only; the first is imported at monolith import time,
    # the second by the skill loader and the web server.
    "core.camera_tiles", "core.web_panels",
    # The NIGHT_QUIET_ENABLED reader: stdlib-only, imported by
    # core.emotion_tracker at import time (core.tts and core.tone_detector
    # import it lazily so both still run as scripts).
    "core.night_quiet",
    # The world clock (v2.0.148): stdlib zoneinfo only, imported at monolith
    # import time (fast paths + the reply guard).
    "core.world_clock",
    # Spoken arithmetic (2026-10-01): stdlib only, imported by the prompt
    # router and the fast paths.
    "core.spoken_math",
    # The speech-queue presence rule (2026-10-02): stdlib only, imported at
    # monolith import time.
    "core.owner_presence",
    # The local prompt budget (2026-10-01): stdlib only, imported at
    # monolith import time.
    "core.prompt_budget",
    # Brain glow (2026-10-02): stdlib only — the HUD subprocesses import its
    # reader, so it must load with nothing but the standard library.
    "core.brain_glow",
    # The retired-model guard (2026-10-02): stdlib only, imported by
    # core.llm_client (every Claude call) and core.orchestrator.
    "core.claude_model_guard",
    # Instant actions (2026-10-02): stdlib + core modules that do no I/O at
    # import, imported at monolith import time.
    "core.instant_actions",
    # A yes to JARVIS's own offer (2026-10-02): stdlib + core.yes_no,
    # imported at monolith import time.
    "core.offer_reply",
    # The verified streaming links, the opened-by-JARVIS ledger and the
    # monitor geometry (2026-10-02 streaming-control fixes): stdlib-only,
    # imported at monolith import time and by core.dispatcher.
    "core.streaming_search", "core.opened_ledger", "core.monitor_geometry",
    # Which windows a window command may touch (2026-10-03): stdlib-only at
    # import (ctypes / psutil lazily), imported by core.actions and at
    # monolith import time.
    "core.window_scope",
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
        # Compiles IN MEMORY. This is a syntax sweep, so it must not write .pyc
        # files into the live repo's __pycache__ -- but it used
        # py_compile.compile(), which does. On a Windows box with on-access
        # antivirus the overwrite of the freshly written ~1.2 MB monolith .pyc
        # was transiently refused (WinError 5: still refused after 50 ms,
        # accepted after 200 ms; no process held the file), which failed this
        # test under tools/run_tests_ci_sim.py on 2026-09-28. compile() on the
        # source bytes is exactly what py_compile does before it writes
        # (importlib's source_to_code), so the check itself is unchanged.
        failures = []
        count = 0
        for path in _iter_source_files():
            count += 1
            try:
                with open(path, "rb") as fh:
                    compile(fh.read(), path, "exec", dont_inherit=True)
            except Exception as exc:  # SyntaxError, bad encoding, null bytes...
                failures.append(f"{os.path.relpath(path, _PROJECT_ROOT)}: "
                                f"{type(exc).__name__}: {exc}")
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
