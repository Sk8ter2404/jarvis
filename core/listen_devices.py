"""core/listen_devices.py — where each listening model runs (2026-10-04).

WHY THIS EXISTS
---------------
JARVIS's ears are five models: Parakeet (the owner's commands, CPU), Smart
Turn (end of turn, CPU), Whisper large-v3-turbo (the rescue + the ambient
listener, on the GTX 1650 "listen card"), Silero (CPU) and Resemblyzer
voice-ID. Where each one ran was decided in five places, one of them by
accident: Resemblyzer's own default put voice-ID on cuda:0 — the RTX 3090
that holds the brain with ~1.1 GB to spare — and it grew a cache there sized
by the longest clip it had seen (measured 2026-10-04: 135 MB for a 2.5 s clip,
441 MB for a 30 s one, never released).

One setting per model now says where it runs, and one resolver turns that into
a device, checks the card is there and has room, and falls back to the CPU
with ONE log line when it is not:

    WHISPER_DEVICE     'auto' | 'cuda' | 'cuda:N' | 'cpu' | 'listen'
    PARAKEET_DEVICE    'cpu' | 'listen' | 'cuda:N'     (default 'cpu')
    SMART_TURN_DEVICE  'cpu' | 'listen' | 'cuda:N'     (default 'cpu')
    VOICE_ID_DEVICE    'cpu' | 'listen' | 'cuda:N'     (default 'cpu')
    LISTEN_GPU         which card 'listen' means: 'cuda:N', a GPU UUID
                       ('GPU-1a2b...', a unique prefix is enough) or a piece
                       of its name ('1650'); '' = no listen card (CPU).
                       Default 'cuda:1', the second card by PCI bus.
    LISTEN_GPU_RESERVE_MB  free VRAM a model must leave on its card (512).

The defaults are what was measured best on this desk (2026-10-04): Parakeet
on the CPU decodes a command in 106-194 ms (no GPU onnxruntime is installed —
onnxruntime-gpu is an owner decision); Smart Turn answers in 25 ms on 4 CPU
threads; voice-ID takes 10-75 ms on the CPU, so a GPU buys nothing for it;
Whisper stays where WHISPER_DEVICE puts it (the owner's 'cuda:1', the 1650).
When a second RTX 3090 arrives, LISTEN_GPU = its UUID (printed on the boot
"[listen] devices:" line) moves every model set to 'listen' at once.

Public API (stdlib only, never raises):
    normalize(spec)                       -> str
    resolve(model, spec, *, listen_gpu, reserve_mb, need_mb, env, probe,
            runtime)                      -> Placement
    resolve_target(spec, *, listen_gpu, env, probe) -> Placement
    ort_cuda_runtime(ort=None)            -> runtime callable
    ort_providers(placement)              -> onnxruntime providers list
    fallback_line(model, placement)       -> str ('' when placed as asked)
"""
from __future__ import annotations

from typing import NamedTuple

from core import gpu_probe as _gpu_probe

DEFAULT_LISTEN_GPU = "cuda:1"
DEFAULT_RESERVE_MB = 512

# What one model needs on a card (MB): weights + workspace, measured or
# estimated on 2026-10-04. Whisper's figures live with its plan in
# bobert_companion (_whisper_cuda_plan); these are the rest.
NEED_MB = {
    "parakeet": 1200,     # 0.6B int8 ONNX (~650 MB) + CUDA EP workspace
    "smart_turn": 300,    # ~9 MB Whisper-tiny encoder + EP workspace
    "voice_id": 400,      # 36 MB weights + torch cache for a 30 s clip (364)
}


class Placement(NamedTuple):
    device: str            # 'cpu' | 'cuda:N'
    index: "int | None"    # the CUDA index on a GPU, else None
    asked: str             # the setting as normalised
    reason: str            # '' = placed as asked; else why it is on the CPU
    gpu: "dict | None"     # core.gpu_probe's record for the card


def normalize(spec) -> str:
    """The setting lower-cased and stripped; '' -> 'cpu'. Never raises."""
    try:
        s = str(spec if spec is not None else "").strip().lower()
    except Exception:
        return "cpu"
    return s or "cpu"


def _cpu(asked: str, reason: str = "", gpu=None) -> Placement:
    return Placement("cpu", None, asked, reason, gpu)


def resolve_target(spec, *, listen_gpu=DEFAULT_LISTEN_GPU, env=None,
                   probe=None) -> Placement:
    """Which card ``spec`` names, without a memory check: 'cpu' as itself,
    'listen' through ``listen_gpu``, 'cuda' / 'cuda:N' directly. A card that
    is missing (or a mapping core.gpu_probe cannot know) is the CPU with a
    reason. Never raises."""
    asked = normalize(spec)
    try:
        probe = probe if probe is not None else _gpu_probe
        if asked == "cpu":
            return _cpu(asked)
        if asked == "listen":
            target = str(listen_gpu or "").strip()
            if not target:
                return _cpu(asked, "LISTEN_GPU is empty (no listen card)")
        elif asked == "cuda" or asked.startswith("cuda:"):
            target = asked
        else:
            return _cpu(asked, f"unknown device {asked!r} (cpu | listen | "
                               f"cuda:N)")
        hit = probe.find(target, env)
        if hit is None:
            label = f"listen card {target!r}" if asked == "listen" else target
            return _cpu(asked, f"{label} not found")
        idx, gpu = hit
        return Placement(f"cuda:{int(idx)}", int(idx), asked, "", gpu)
    except Exception as e:
        return _cpu(asked, f"device check failed ({type(e).__name__})")


def resolve(model: str, spec, *, listen_gpu=DEFAULT_LISTEN_GPU,
            reserve_mb=DEFAULT_RESERVE_MB, need_mb=None, env=None,
            probe=None, runtime=None) -> Placement:
    """Where ``model`` runs for setting ``spec``: resolve_target, then
    ``runtime(cuda_index) -> (ok, why)`` (the model's own GPU runtime is
    there, e.g. onnxruntime's CUDA provider), then free VRAM >= ``need_mb``
    (NEED_MB[model] when None) + ``reserve_mb``. Any failure = the CPU, with
    the reason for the one log line. Never raises."""
    p = resolve_target(spec, listen_gpu=listen_gpu, env=env, probe=probe)
    if p.index is None:
        return p
    try:
        if runtime is not None:
            try:
                ok, why = runtime(p.index)
            except Exception as e:
                ok, why = False, f"runtime check failed ({type(e).__name__})"
            if not ok:
                return _cpu(p.asked, why or "no GPU runtime", p.gpu)
        need = NEED_MB.get(model, 0) if need_mb is None else need_mb
        try:
            need = max(0, int(need))
        except Exception:
            need = 0
        try:
            reserve = max(0, int(reserve_mb))
        except Exception:
            reserve = DEFAULT_RESERVE_MB
        free = (p.gpu or {}).get("free_mb")
        if need or reserve:
            if free is None:
                return _cpu(p.asked, f"{p.device} free VRAM unreadable",
                            p.gpu)
            if int(free) < need + reserve:
                return _cpu(p.asked, f"{p.device} has {int(free)} MB free < "
                                     f"{need} + {reserve} MB reserve", p.gpu)
        return p
    except Exception as e:
        return _cpu(p.asked, f"device check failed ({type(e).__name__})",
                    p.gpu)


def ort_cuda_runtime(ort=None):
    """``runtime`` for an onnxruntime model: ok only when this onnxruntime
    build offers CUDAExecutionProvider (the plain 'onnxruntime' wheel —
    the one installed here — offers CPU only). ``ort`` = the module (tests
    pass a fake); imported lazily otherwise."""
    def _check(_index):
        try:
            mod = ort
            if mod is None:
                import onnxruntime as mod  # noqa: F811 - lazy, CI has none
            provs = list(mod.get_available_providers())
        except Exception as e:
            return False, f"onnxruntime unavailable ({type(e).__name__})"
        if "CUDAExecutionProvider" in provs:
            return True, ""
        return False, ("onnxruntime has no CUDA provider (CPU build; "
                       "onnxruntime-gpu is not installed)")
    return _check


def ort_providers(placement) -> list:
    """onnxruntime ``providers=`` for a placement: CUDA on its card with the
    CPU behind it (ops CUDA cannot run fall back per node), or CPU only."""
    try:
        if placement is not None and placement.index is not None:
            return [("CUDAExecutionProvider",
                     {"device_id": int(placement.index)}),
                    "CPUExecutionProvider"]
    except Exception:
        pass
    return ["CPUExecutionProvider"]


def fallback_line(model: str, placement) -> str:
    """The one log line for a model that asked for a card and got the CPU
    ('' when it runs where it asked, or asked for the CPU)."""
    try:
        if placement is None or not placement.reason:
            return ""
        return (f"  [listen] {model}: asked {placement.asked} — "
                f"{placement.reason}; running on the CPU")
    except Exception:
        return ""


def label(placement) -> str:
    """'cpu' or 'cuda:1 (NVIDIA GeForce GTX 1650 SUPER)' for the boot
    line."""
    try:
        if placement is None:
            return "?"
        if placement.index is None:
            return "cpu"
        name = str((placement.gpu or {}).get("name") or "").strip()
        return f"{placement.device} ({name})" if name else placement.device
    except Exception:
        return "?"
