"""core/cuda_preinit.py - start the NVIDIA CUDA driver BEFORE ctranslate2 loads.

WHY THIS EXISTS (the v2.0.179 crash, 2026-10-04 20:40)
------------------------------------------------------
v2.0.179 died 2 1/2 minutes after boot with "Fatal Python error: Aborted". The
crash dumps (LocalDumps, pythonw.exe.158452.dmp) show the faulting thread was
the self-diagnostic's short-lived ``probe-stt`` thread, EXITING after its
one Whisper decode on cuda:1: the Windows loader ran ctranslate2.dll's
thread-local destructors (CTranslate2 keeps a CUDA stream / cuBLAS / cuDNN
handle per host thread), one of them called ``ctranslate2::set_device_index``
-> ``cudaSetDevice``, the driver answered "No CUDA context is current to the
calling thread" (CUDA_ERROR_INVALID_CONTEXT, 201), CTranslate2 threw
``std::runtime_error`` from a destructor, and the process went
std::terminate -> abort.

v2.0.178 ran the same probe on a fresh thread 94 times that day without a
crash. The difference is DLL ORDER. When a thread exits, Windows runs every
DLL's thread-detach code (TLS callbacks + DllMain) in REVERSE initialisation
order, under the loader lock. 178's Whisper plan called
``torch.cuda.mem_get_info()`` first, so the CUDA driver (nvcuda64.dll, module
#156) was initialised long before ctranslate2.dll (#252): on thread exit
CTranslate2's destructors ran while the driver still knew the thread. 179
reads free VRAM through NVML instead (no CUDA context - that part is right),
so ctranslate2.dll (#200) loaded first and the driver (#263) only at the first
CUDA call: the driver now cleaned the dying thread up FIRST, and CTranslate2's
destructors found nothing to talk to.

WHAT THIS MODULE DOES
---------------------
``before_ctranslate2()`` loads the driver library and calls ``cuInit(0)`` -
once per process, and before anything imports ctranslate2 / faster_whisper.
``cuInit`` initialises the driver API only: it creates NO context on any GPU
and allocates no VRAM (contexts come from cuCtxCreate /
cuDevicePrimaryCtxRetain, which this module never calls), so the 3090 stays
untouched exactly as 179 intended. It restores 178's proven order without
torch.

If ctranslate2 is ALREADY loaded when it runs, the order can no longer be
fixed; that is reported (``late``) so the boot log says so. The second line of
defence - every CTranslate2 call on a CUDA model runs on ONE thread that never
exits (core/ct2_host.py) - covers that case and any other thread-exit path.

``load_order()`` reads this process's module list (psapi EnumProcessModules,
load order) so boot can log whether nvcuda64.dll really sits before
ctranslate2.dll - the exact measurement the investigation made by hand.

Never raises. Skipped (nothing loaded) when CUDA_VISIBLE_DEVICES hides every
GPU ("" or a negative first entry): no GPU can be used then, so there is no
CUDA thread state to protect. Also skipped under JARVIS_TEST_MODE=1 or
JARVIS_NO_CUDA_DRIVER=1 (tests/__init__.py sets it): a test run never loads
the real driver.
"""
from __future__ import annotations

import os
import sys
import threading

_lock = threading.Lock()
_state: dict = {"done": False, "ran": False, "ok": None, "rc": None,
                "late": None, "detail": "not run"}
_keep: list = [None]          # the driver library handle, kept for the process


def _default_loader():
    """The CUDA driver library (nvcuda.dll / libcuda.so.1); raises OSError
    when no NVIDIA driver is installed."""
    import ctypes
    if os.name == "nt":
        return ctypes.WinDLL("nvcuda.dll")
    return ctypes.CDLL("libcuda.so.1")


def _default_ct2_loaded() -> bool:
    """True when ctranslate2 is already in this process (its Python package
    imported, or its DLL loaded by anything). Never raises."""
    try:
        if "ctranslate2" in sys.modules:
            return True
        if os.name == "nt":
            import ctypes
            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            k32.GetModuleHandleW.restype = ctypes.c_void_p
            k32.GetModuleHandleW.argtypes = [ctypes.c_wchar_p]
            return bool(k32.GetModuleHandleW("ctranslate2.dll"))
    except Exception:
        pass
    return False


_loader = _default_loader
_ct2_loaded = _default_ct2_loaded


def set_hooks(loader=None, ct2_loaded=None) -> None:
    """Tests: replace the driver loader / the "is ctranslate2 loaded" probe
    and forget any earlier run. None restores the real one."""
    global _loader, _ct2_loaded
    with _lock:
        _loader = loader if loader is not None else _default_loader
        _ct2_loaded = ct2_loaded if ct2_loaded is not None else _default_ct2_loaded
        _keep[0] = None
        _state.clear()
        _state.update({"done": False, "ran": False, "ok": None, "rc": None,
                       "late": None, "detail": "not run"})


def _no_visible_gpu(env) -> bool:
    """CUDA_VISIBLE_DEVICES set so that CUDA sees no device: "" or a first
    entry that is a negative number (CUDA stops at the first bad entry)."""
    if "CUDA_VISIBLE_DEVICES" not in env:
        return False
    vis = str(env.get("CUDA_VISIBLE_DEVICES") or "").strip()
    if not vis:
        return True
    first = vis.split(",")[0].strip()
    try:
        return int(first) < 0
    except ValueError:
        return False


def before_ctranslate2(env=None) -> dict:
    """Initialise the CUDA driver (cuInit(0), no context) once per process,
    before ctranslate2 loads. Idempotent and thread-safe; returns a copy of
    the outcome: done, ran (cuInit was called), ok, rc, late (ctranslate2
    was already loaded, so the order could not be fixed), detail. Never
    raises."""
    try:
        env = os.environ if env is None else env
        with _lock:
            if _state["done"]:
                return dict(_state)
            if str(env.get("JARVIS_TEST_MODE", "")) == "1":
                _state.update(done=True, detail="skipped: JARVIS_TEST_MODE=1")
                return dict(_state)
            if str(env.get("JARVIS_NO_CUDA_DRIVER", "")) == "1":
                # Set by tests/__init__.py: a unit test never loads the real
                # driver.
                _state.update(done=True, detail="skipped: JARVIS_NO_CUDA_DRIVER=1")
                return dict(_state)
            if _no_visible_gpu(env):
                _state.update(done=True, detail="skipped: CUDA_VISIBLE_DEVICES "
                              "hides every GPU")
                return dict(_state)
            late = bool(_ct2_loaded())
            try:
                lib = _loader()
            except Exception as e:
                _state.update(done=True, ok=False, late=late,
                              detail=f"no CUDA driver ({type(e).__name__})")
                return dict(_state)
            try:
                rc = int(lib.cuInit(0))
            except Exception as e:
                _state.update(done=True, ok=False, late=late,
                              detail=f"cuInit raised {type(e).__name__}: {e}")
                return dict(_state)
            _keep[0] = lib
            _state.update(done=True, ran=True, ok=(rc == 0), rc=rc, late=late,
                          detail=("cuInit(0) = 0" if rc == 0
                                  else f"cuInit(0) returned {rc}"))
            return dict(_state)
    except Exception as e:     # pragma: no cover - defensive
        return {"done": False, "ran": False, "ok": False, "rc": None,
                "late": None, "detail": f"{type(e).__name__}: {e}"}


def state() -> dict:
    """A copy of the last outcome (see before_ctranslate2)."""
    with _lock:
        return dict(_state)


def status_line(st=None) -> str:
    """One "[cuda-init] ..." line for the boot log. Never raises."""
    try:
        st = state() if st is None else st
        if not st.get("ran"):
            return f"  [cuda-init] {st.get('detail', 'not run')}"
        if st.get("late"):
            return ("  [cuda-init] WARNING: ctranslate2 was already loaded "
                    f"when the CUDA driver started ({st.get('detail')}) - a "
                    "thread that used it may abort the process when it "
                    "exits; CTranslate2 CUDA work stays on the ct2-host "
                    "thread")
        if st.get("ok"):
            return ("  [cuda-init] CUDA driver started before ctranslate2 "
                    "(cuInit(0) = 0, no context created)")
        return f"  [cuda-init] CUDA driver not started: {st.get('detail')}"
    except Exception:          # pragma: no cover - defensive
        return "  [cuda-init] status unavailable"


# ─── load-order check (boot log) ──────────────────────────────────────────

def _default_module_names() -> "list[str] | None":
    """This process's module base names in LOAD order (psapi), lower-case;
    None off Windows or when the list cannot be read. Read-only: no loader
    lock is taken, nothing is loaded."""
    if os.name != "nt":
        return None
    try:
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        proc = k32.GetCurrentProcess
        proc.restype = wintypes.HANDLE
        enum = k32.K32EnumProcessModules
        enum.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.HMODULE),
                         wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
        enum.restype = wintypes.BOOL
        base = k32.K32GetModuleBaseNameW
        base.argtypes = [wintypes.HANDLE, wintypes.HMODULE, wintypes.LPWSTR,
                         wintypes.DWORD]
        base.restype = wintypes.DWORD
        h = proc()
        n = 1024
        for _ in range(3):
            arr = (wintypes.HMODULE * n)()
            need = wintypes.DWORD(0)
            if not enum(h, arr, ctypes.sizeof(arr), ctypes.byref(need)):
                return None
            count = need.value // ctypes.sizeof(wintypes.HMODULE)
            if count <= n:
                break
            n = count + 64
        else:
            return None
        out = []
        buf = ctypes.create_unicode_buffer(260)
        for i in range(count):
            if not arr[i]:
                continue
            ln = base(h, arr[i], buf, 260)
            out.append(buf.value[:ln].lower() if ln else "")
        return out
    except Exception:
        return None


_module_names = _default_module_names


def load_order(names=None) -> dict:
    """Where the CUDA driver and ctranslate2 sit in the process's module list
    (1-based load positions, None when absent) and whether the driver comes
    first: {"nvcuda": n, "nvcuda64": n, "ctranslate2": n, "driver_first":
    True | False | None}. Never raises."""
    out = {"nvcuda": None, "nvcuda64": None, "ctranslate2": None,
           "driver_first": None}
    try:
        names = _module_names() if names is None else names
        if not names:
            return out
        for i, nm in enumerate(names, start=1):
            key = {"nvcuda.dll": "nvcuda", "nvcuda64.dll": "nvcuda64",
                   "ctranslate2.dll": "ctranslate2"}.get(nm)
            if key and out[key] is None:
                out[key] = i
        drv = [p for p in (out["nvcuda"], out["nvcuda64"]) if p is not None]
        if out["ctranslate2"] is not None and drv:
            out["driver_first"] = max(drv) < out["ctranslate2"]
        return out
    except Exception:
        return out


def load_order_line(order=None) -> str:
    """One "[cuda-init] load order ..." line. Never raises."""
    try:
        o = load_order() if order is None else order
        if o.get("ctranslate2") is None:
            return "  [cuda-init] load order: ctranslate2 not loaded"
        pos = (f"nvcuda #{o.get('nvcuda')}, nvcuda64 #{o.get('nvcuda64')}, "
               f"ctranslate2 #{o.get('ctranslate2')}")
        if o.get("driver_first") is True:
            return f"  [cuda-init] load order OK: {pos} (driver first)"
        if o.get("driver_first") is False:
            return (f"  [cuda-init] WARNING load order: {pos} - ctranslate2 "
                    "loaded before the CUDA driver")
        return f"  [cuda-init] load order: {pos}"
    except Exception:          # pragma: no cover - defensive
        return "  [cuda-init] load order unavailable"
