"""core/gpu_probe.py — free VRAM per GPU from NVML, WITHOUT a CUDA context.

WHY THIS EXISTS
---------------
Every free-VRAM check in JARVIS used ``torch.cuda.mem_get_info(i)``. That call
CREATES a CUDA context on GPU ``i`` the first time it runs (measured
2026-10-04 on this box's torch 2.10+cu126: +59 MB on the card for the context
alone, plus the cuBLAS/cuDNN state that follows it), and the context lives
until the process exits. Four copies asked GPU 0 — the RTX 3090 that holds the
local brain with ~1.1 GB to spare — so merely ASKING "is there room?" took
room. (The 3090 context the investigation actually caught live was Resemblyzer
defaulting to "cuda"; core/voice_id.py now places it explicitly. These probes
were the latent half of the same hazard.)

NVML — the library nvidia-smi is built on, shipped with every NVIDIA driver
(``nvml.dll`` in System32 / ``libnvidia-ml.so.1``) — reads the same counters
from the driver without a context. This module loads it through ctypes (no
pynvml package: it is not installed here, which made
skills/standby_audio_detect's NVML probe dead code), initialises it once for
the process (~30 ms) and answers in well under a millisecond after that.

CUDA INDEX vs NVML INDEX
------------------------
NVML numbers GPUs by PCI bus. CUDA numbers them by CUDA_DEVICE_ORDER
(FASTEST_FIRST by default) after CUDA_VISIBLE_DEVICES filters them. The two
agree only under ``CUDA_DEVICE_ORDER=PCI_BUS_ID`` — bobert_companion sets that
before anything touches CUDA, so these reads index the same cards ctranslate2
and torch use. It does NOT make an index an identity: a card added on a lower
PCI bus than the 1650 (bus 8 here; a CPU-attached slot is lower) takes
"cuda:1" and the 1650 becomes "cuda:2" — under either order. Settings that
must stay on one card name it (its UUID, or "1650"; see find()).
``cuda_gpus()`` maps CUDA's numbering onto NVML's and returns None — "unknown",
never a guess — when it cannot (FASTEST_FIRST with more than one GPU, an
unreadable CUDA_VISIBLE_DEVICES entry). A GPU NVML cannot read keeps its
place as an UNREADABLE entry (name '', free_mb None), so one bad card never
shifts the cards after it onto its index (review 2026-10-04: a skipped 3090
made the 1650's numbers read as cuda:0's).

Public API (stdlib only, never raises, never creates a CUDA context):
    available()                 -> bool        NVML loaded and initialised
    gpus()                      -> list[dict]  every GPU, NVML (PCI) order
    cuda_gpus(env=None)         -> list|None   the GPUs CUDA numbers 0, 1, ...
    cuda_gpu(index, env=None)   -> dict|None
    cuda_memory_mb(index, env=None) -> (free_mb, total_mb) | None
    find(spec, env=None)        -> (cuda_index, dict) | None
    describe(gpu)               -> str
Each GPU dict: index (NVML), name, uuid, pci_bus, total_mb, free_mb,
used_mb, util_pct (None when unreadable). cuda_gpus() also holds an
UNREADABLE entry (``unreadable: True``, every reading None) for a GPU NVML
counts but cannot read; gpus() leaves those out.
"""
from __future__ import annotations

import ctypes
import os
import sys
import threading

NVML_SUCCESS = 0
_NAME_LEN = 96          # NVML_DEVICE_NAME_V2_BUFFER_SIZE
_UUID_LEN = 96          # NVML_DEVICE_UUID_V2_BUFFER_SIZE
_MB = 1024 * 1024


class _Memory(ctypes.Structure):            # nvmlMemory_t
    _fields_ = [("total", ctypes.c_ulonglong),
                ("free", ctypes.c_ulonglong),
                ("used", ctypes.c_ulonglong)]


class _Utilization(ctypes.Structure):       # nvmlUtilization_t
    _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]


class _PciInfo(ctypes.Structure):           # nvmlPciInfo_t
    _fields_ = [("busIdLegacy", ctypes.c_char * 16),
                ("domain", ctypes.c_uint), ("bus", ctypes.c_uint),
                ("device", ctypes.c_uint), ("pciDeviceId", ctypes.c_uint),
                ("pciSubSystemId", ctypes.c_uint),
                ("busId", ctypes.c_char * 32)]


def _candidate_paths() -> "list[str]":
    if sys.platform == "win32" or os.name == "nt":
        windir = os.environ.get("WINDIR", r"C:\Windows")
        return ["nvml.dll",
                os.path.join(windir, "System32", "nvml.dll"),
                r"C:\Program Files\NVIDIA Corporation\NVSMI\nvml.dll"]
    return ["libnvidia-ml.so.1", "libnvidia-ml.so"]


def _default_loader():
    """The NVML library, or None when no driver ships one."""
    for path in _candidate_paths():
        try:
            return ctypes.CDLL(path)
        except OSError:
            continue
        except Exception:
            continue
    return None


_loader = _default_loader     # tests: set_loader()
_state = {"lib": None, "tried": False}
_lock = threading.Lock()


def set_loader(fn) -> None:
    """Replace the library loader (tests pass a fake NVML) and forget the
    loaded library, so the next call loads through ``fn``."""
    global _loader
    with _lock:
        _loader = fn if fn is not None else _default_loader
        _state["lib"] = None
        _state["tried"] = False


def _lib():
    """The initialised NVML library, loaded and initialised once per process;
    None when it is missing or nvmlInit fails (then never retried)."""
    if _state["tried"]:
        return _state["lib"]
    with _lock:
        if _state["tried"]:
            return _state["lib"]
        lib = None
        try:
            cand = _loader()
            if cand is not None and int(cand.nvmlInit_v2()) == NVML_SUCCESS:
                lib = cand
        except Exception:
            lib = None
        _state["lib"] = lib
        _state["tried"] = True
        return lib


def available() -> bool:
    """NVML is loaded and initialised. Never raises."""
    try:
        return _lib() is not None
    except Exception:
        return False


def _text(buf) -> str:
    try:
        raw = buf.value
        return raw.decode("utf-8", "replace") if isinstance(raw, bytes) \
            else str(raw)
    except Exception:
        return ""


def _read_one(lib, i: int) -> "dict | None":
    h = ctypes.c_void_p()
    if int(lib.nvmlDeviceGetHandleByIndex_v2(ctypes.c_uint(i),
                                             ctypes.byref(h))) != NVML_SUCCESS:
        return None
    mem = _Memory()
    if int(lib.nvmlDeviceGetMemoryInfo(h, ctypes.byref(mem))) != NVML_SUCCESS:
        return None
    name = ctypes.create_string_buffer(_NAME_LEN)
    uuid = ctypes.create_string_buffer(_UUID_LEN)
    try:
        lib.nvmlDeviceGetName(h, name, ctypes.c_uint(_NAME_LEN))
    except Exception:
        pass
    try:
        lib.nvmlDeviceGetUUID(h, uuid, ctypes.c_uint(_UUID_LEN))
    except Exception:
        pass
    pci_bus = None
    try:
        pci = _PciInfo()
        if int(lib.nvmlDeviceGetPciInfo_v3(h, ctypes.byref(pci))) \
                == NVML_SUCCESS:
            pci_bus = int(pci.bus)
    except Exception:
        pci_bus = None
    util = None
    try:
        u = _Utilization()
        if int(lib.nvmlDeviceGetUtilizationRates(h, ctypes.byref(u))) \
                == NVML_SUCCESS:
            util = int(u.gpu)
    except Exception:
        util = None
    return {"index": int(i), "name": _text(name), "uuid": _text(uuid),
            "pci_bus": pci_bus, "total_mb": int(mem.total) // _MB,
            "free_mb": int(mem.free) // _MB, "used_mb": int(mem.used) // _MB,
            "util_pct": util}


def _unreadable(i: int) -> dict:
    """The place-holder for NVML index ``i`` when that GPU cannot be read:
    it keeps the GPUs after it on their own indices, and matches no name or
    UUID."""
    return {"index": int(i), "name": "", "uuid": "", "pci_bus": None,
            "total_mb": None, "free_mb": None, "used_mb": None,
            "util_pct": None, "unreadable": True}


def _slots() -> "list[dict]":
    """One entry per GPU NVML counts, in NVML (PCI bus) order — the GPU's
    record, or _unreadable(i) when it cannot be read; [] when NVML is
    unavailable or counts nothing. Never raises."""
    try:
        lib = _lib()
        if lib is None:
            return []
        n = ctypes.c_uint(0)
        if int(lib.nvmlDeviceGetCount_v2(ctypes.byref(n))) != NVML_SUCCESS:
            return []
        out = []
        for i in range(int(n.value)):
            try:
                g = _read_one(lib, i)
            except Exception:
                g = None
            out.append(g if g is not None else _unreadable(i))
        return out
    except Exception:
        return []


def gpus() -> "list[dict]":
    """Every GPU the driver reports that NVML can read, in NVML (PCI bus)
    order; [] when NVML is unavailable or reads nothing. For display: the
    list POSITION is not a CUDA index (cuda_gpus() is). Never raises."""
    return [g for g in _slots() if not g.get("unreadable")]


def cuda_gpus(env=None) -> "list[dict] | None":
    """The GPUs as CUDA numbers them in a process with environment ``env``
    (default os.environ): element N is ``cuda:N``. None when the mapping
    cannot be known — NVML unavailable, CUDA_DEVICE_ORDER left at
    FASTEST_FIRST with more than one GPU, or a CUDA_VISIBLE_DEVICES entry
    that names nothing. ``CUDA_VISIBLE_DEVICES=""`` = no GPU ([]). Never
    raises."""
    try:
        env = os.environ if env is None else env
        if not available():
            return None
        all_gpus = _slots()    # unreadable GPUs keep their place
        order = str(env.get("CUDA_DEVICE_ORDER", "") or "").strip().upper()
        pci_order = order == "PCI_BUS_ID" or len(all_gpus) <= 1
        vis = env.get("CUDA_VISIBLE_DEVICES")
        if vis is None:
            return list(all_gpus) if pci_order else None
        if not str(vis).strip():
            return []
        out = []
        for tok in str(vis).split(","):
            tok = tok.strip()
            if not tok:
                break
            if tok.lstrip("-").isdigit():
                i = int(tok)
                if i < 0:
                    break                 # CUDA stops at the first bad entry
                if not pci_order:
                    return None
                if i >= len(all_gpus):
                    break
                out.append(all_gpus[i])
                continue
            hit = [g for g in all_gpus
                   if g.get("uuid") and g["uuid"].lower().startswith(
                       tok.lower() if tok.lower().startswith("gpu-")
                       else "gpu-" + tok.lower())]
            if len(hit) != 1:
                return None
            out.append(hit[0])
        return out
    except Exception:
        return None


def cuda_gpu(index, env=None) -> "dict | None":
    """The GPU CUDA calls ``cuda:index``, or None (unknown / absent)."""
    try:
        i = int(index)
        lst = cuda_gpus(env)
        if lst is None or not 0 <= i < len(lst):
            return None
        return lst[i]
    except Exception:
        return None


def cuda_memory_mb(index, env=None) -> "tuple[int, int] | None":
    """(free_mb, total_mb) of ``cuda:index``, or None when unknown (no such
    GPU, or NVML cannot read it). Never raises, never creates a CUDA
    context."""
    g = cuda_gpu(index, env)
    if g is None or g.get("unreadable"):
        return None
    try:
        return int(g["free_mb"]), int(g["total_mb"])
    except Exception:
        return None


def find(spec, env=None) -> "tuple[int, dict] | None":
    """The GPU a placement setting names, as ``(cuda_index, gpu)``; None
    when nothing matches or the mapping is unknown. ``spec``: "cuda" (=
    cuda:0), "cuda:N", a GPU UUID ("GPU-1a2b..." — a unique prefix is
    enough), or a case-insensitive piece of the name ("1650", "3090" — the
    first match in CUDA order). Never raises."""
    try:
        s = str(spec or "").strip()
        if not s:
            return None
        lst = cuda_gpus(env)
        if not lst:
            return None
        low = s.lower()
        if low == "cuda":
            return 0, lst[0]
        if low.startswith("cuda:"):
            try:
                i = int(low.split(":", 1)[1])
            except ValueError:
                return None
            return (i, lst[i]) if 0 <= i < len(lst) else None
        if low.startswith("gpu-") or low.startswith("mig-"):
            hits = [(i, g) for i, g in enumerate(lst)
                    if str(g.get("uuid", "")).lower().startswith(low)]
            if len(hits) != 1:
                return None
            # A SHORT prefix might also be an unreadable GPU's: unknown. A
            # whole UUID ("GPU-" + 36) that matched a readable GPU is that
            # GPU, whatever else cannot be read.
            if len(low) < 40 and any(g.get("unreadable") for g in lst):
                return None
            return hits[0]
        for i, g in enumerate(lst):
            if low in str(g.get("name", "")).lower():
                return i, g
        return None
    except Exception:
        return None


def describe(gpu, cuda_index=None) -> str:
    """One short phrase for a log line: 'cuda:1 NVIDIA GeForce GTX 1650
    SUPER (bus 8, GPU-1a2b3c4d) 2100/4096 MB free'. Never raises."""
    try:
        if not gpu:
            return "no GPU"
        where = f"cuda:{cuda_index} " if cuda_index is not None else ""
        uuid = str(gpu.get("uuid") or "")
        extra = []
        if gpu.get("pci_bus") is not None:
            extra.append(f"bus {gpu['pci_bus']}")
        if uuid:
            extra.append(uuid[:12])
        tail = f" ({', '.join(extra)})" if extra else ""
        if gpu.get("unreadable") or gpu.get("free_mb") is None:
            return f"{where}{gpu.get('name') or 'GPU'}{tail} unreadable"
        return (f"{where}{gpu.get('name') or 'GPU'}{tail} "
                f"{int(gpu.get('free_mb', 0))}/{int(gpu.get('total_mb', 0))}"
                f" MB free")
    except Exception:
        return "GPU"
