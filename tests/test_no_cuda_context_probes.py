"""Ratchet: nothing in JARVIS may open a CUDA context just to look (2026-10-04).

THE LIVE FINDING: the running JARVIS held ~755 MB on the RTX 3090 (the brain's
card, ~1.1 GB to spare). Two code shapes create a CUDA context on a card as a
side effect:

  * ``torch.cuda.mem_get_info(i)`` — the free-VRAM probe. Four copies asked
    GPU 0 (bobert_companion._cuda0_free_vram_mb, core/voice_clone
    _free_vram_ok) or the 1650 (_whisper_cuda_plan; skills/standby_audio_detect
    used pynvml, which is not installed). Replayed on the 1650: +59 MB.
  * ``VoiceEncoder()`` with no device — Resemblyzer picks "cuda" = cuda:0
    (the live 3090 context: born inside the ambient listener's first
    voice-ID, 13:50:56.037).

Free VRAM is read through core/gpu_probe.py (NVML, no context) and every
VoiceEncoder is built on an explicit device. This walks the AST of every
production file (outside tests/), so a comment or a docstring that NAMES the
old call never counts — only a real call does.

    python -m unittest tests.test_no_cuda_context_probes
"""
from __future__ import annotations

import ast
import os
import unittest

_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SKIP_DIRS = {"tests", ".git", "__pycache__", "venv", ".venv", "backups",
              "node_modules", "data", "data_staging", "logs", "build",
              "dist", ".claude"}


def _production_files():
    for root, dirs, files in os.walk(_PROJECT):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS
                   and not d.startswith(".")]
        for fn in files:
            if fn.endswith(".py"):
                yield os.path.join(root, fn)


def _calls(path):
    try:
        with open(path, encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=path)
    except (SyntaxError, UnicodeDecodeError, OSError):
        return
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            yield node


def _name(func) -> str:
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return ""


class NoContextProbeTests(unittest.TestCase):
    def test_nothing_calls_torch_mem_get_info(self):
        hits = []
        for path in _production_files():
            for call in _calls(path):
                if _name(call.func) == "mem_get_info":
                    hits.append(f"{os.path.relpath(path, _PROJECT)}:"
                                f"{call.lineno}")
        self.assertEqual(hits, [], "torch.cuda.mem_get_info creates a CUDA "
                         "context on the card it asks — read free VRAM "
                         "through core/gpu_probe (NVML) instead")

    def test_every_voice_encoder_names_its_device(self):
        found, bare = [], []
        for path in _production_files():
            for call in _calls(path):
                if _name(call.func) != "VoiceEncoder":
                    continue
                where = f"{os.path.relpath(path, _PROJECT)}:{call.lineno}"
                found.append(where)
                if not (call.args or any(k.arg == "device"
                                         for k in call.keywords)):
                    bare.append(where)
        # Blindness floor: the walk must SEE core/voice_id's constructors.
        self.assertTrue(any(w.replace("\\", "/").startswith("core/voice_id.py")
                            for w in found), found)
        self.assertEqual(bare, [], "VoiceEncoder() with no device lands on "
                         "cuda:0 (the brain's 3090) whenever torch has CUDA")

    def test_the_probes_go_through_gpu_probe(self):
        # The four former mem_get_info / pynvml sites each name gpu_probe.
        want = {
            "bobert_companion.py": ("_cuda0_free_vram_mb",
                                    "_whisper_cuda_plan"),
            os.path.join("core", "voice_clone.py"): ("_free_vram_ok",),
            os.path.join("skills", "standby_audio_detect.py"):
                ("_cuda_free_vram_mb",),
        }
        for rel, fns in want.items():
            with open(os.path.join(_PROJECT, rel), encoding="utf-8") as f:
                tree = ast.parse(f.read())
            defs = {n.name: n for n in ast.walk(tree)
                    if isinstance(n, ast.FunctionDef)}
            for fn in fns:
                # Code only: a docstring naming the old way never counts.
                called, imported = set(), set()
                for n in ast.walk(defs[fn]):
                    if isinstance(n, ast.Call):
                        called.add(_name(n.func))
                    elif isinstance(n, ast.Import):
                        imported.update(a.name for a in n.names)
                    elif isinstance(n, ast.ImportFrom):
                        imported.add(n.module or "")
                self.assertIn("cuda_memory_mb", called, f"{rel}:{fn}")
                self.assertNotIn("mem_get_info", called, f"{rel}:{fn}")
                self.assertNotIn("pynvml", imported, f"{rel}:{fn}")
                self.assertNotIn("torch", imported, f"{rel}:{fn}")


if __name__ == "__main__":
    unittest.main()
