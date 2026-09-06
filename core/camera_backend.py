"""One place that decides WHICH capture backend opens a camera, and at WHICH
index — because on Windows those two questions are not independent.

────────────────────────────────────────────────────────────────────────────
WHY THIS MODULE EXISTS
────────────────────────────────────────────────────────────────────────────
``cv2.VideoCapture(idx, cv2.CAP_DSHOW)`` leaks. Measured on this rig
2026-09-05, 25 open/read/release cycles per figure, each in a fresh process,
1280x720 requested, 30 reads per cycle:

    eMeet C960   CAP_DSHOW   +4.58 OS threads  +479 handles   per cycle
    eMeet C960   CAP_MSMF    -0.21 OS threads    +0.97 handles per cycle
    USB 2.0 Cam  CAP_DSHOW   +4.55 OS threads  +479 handles   per cycle
    USB 2.0 Cam  CAP_MSMF    -0.18 OS threads    +1.06 handles per cycle

The DirectShow threads are all owned by mfksproxy.dll and are never reclaimed;
that is the same leak v2.0.101 gated in the ENUMERATION path, showing up again
in the OPEN path where no gate can help. A DEAD index costs +103 handles under
DirectShow and +0 under Media Foundation, so an index sweep pays it too.

────────────────────────────────────────────────────────────────────────────
THE TRAP: THE TWO BACKENDS DO NOT AGREE ON WHAT AN INDEX MEANS
────────────────────────────────────────────────────────────────────────────
Enumerated on this rig 2026-09-05 (pygrabber for DirectShow,
MFEnumDeviceSources for Media Foundation — the exact calls OpenCV subscripts):

    DirectShow  0=USB 2.0 Camera  1=Kinect V2 Video Sensor  2=eMeet C960  3=OBS Virtual Camera
    MediaFdn.   0=Kinect V2 Video Sensor  1=USB 2.0 Camera  2=eMeet C960

They are NOT the same list and NOT the same length: OBS Virtual Camera has no
Media Foundation presence at all. Swapping the backend constant at a call site
WITHOUT translating the index therefore silently repoints index 0 from the USB
webcam to the Kinect. Confirmed empirically as well as by enumeration: probing
each (backend, index) pair with resolutions only one device on this rig
supports (2592x1944 → USB 2.0 Camera only; 1280x960 → eMeet only; a single
fixed 1920x1080 YUY2 format → Kinect only) reproduces exactly that mapping.

So: this module translates, and it translates BY NAME, which is also the only
form of the question that survives a USB re-enumeration shuffle.

Translating is free. MFEnumDeviceSources measured **+0.000 OS threads and
+0.000 handles per call** over 40 calls, against the DirectShow enumeration's
+0.924 threads / +95.3 handles per call in the same harness on the same boot.
It is also 35x CHEAPER in wall time: 1.2 ms per call against DirectShow's
41.8 ms (medians, n=20 and n=10, timed without the harness around them - an
earlier note here said 107 ms for the MF call, which was this file's own
measurement harness, not the call). That is why this module may enumerate on a
TTL with no fingerprint gate, while the DirectShow enumerator in the monolith
needs one.

This module NEVER calls the DirectShow enumerator itself — the monolith owns
that (gated) primitive and passes its result in, so there is one leaky call
site, not two.

────────────────────────────────────────────────────────────────────────────
THE OTHER TRAP: MSMF "OPENS" A CAMERA SOMEONE ELSE IS HOLDING
────────────────────────────────────────────────────────────────────────────
Measured 2026-09-05, one process holding the eMeet and reading, a second
process then opening the same device:

    contender CAP_DSHOW : isOpened() == False after 0.65 s      (honest refusal)
    contender CAP_MSMF  : isOpened() == True, set(W/H) == True,
                          get(W/H) reads back 1280x720,
                          and 0 of 20 reads produce a frame.

Both backends leave the HOLDER undisturbed (730-742 reads, 0 failures, in
every combination including cross-backend). But an MSMF caller that trusts
``isOpened()`` gets a handle that will never produce an image. That is
precisely the failure this project has paid for before, so
:func:`open_camera` takes ``require_frame`` and callers that need a usable
stream must pass it.

⚠ THAT PARAGRAPH IS ABOUT TWO PROCESSES, AND IT DOES NOT GENERALISE. It used
to end "…so a partial migration is safe". That conclusion was wrong, and it
cost the owner his primary vision every 30 minutes for a day. In-process —
which is the ONLY configuration JARVIS actually runs in, one producer thread
and one prober thread inside the same interpreter — MSMF does NOT refuse the
contender and does NOT leave the holder undisturbed. Measured 2026-09-06,
JARVIS stopped, producer holding MSMF index 1 at 1280x720, 30 s windows either
side, arms identical but for one extra open:

    CONTROL   before 828 reads / 0 fail / 30.0 fps
              during 361 / 0 / 30.1     after 539 / 0 / 30.0
    CONTENDER before 826 reads / 0 fail / 30.0 fps
              during 3,704,096 FAILED reads / 4.4 fps
              after  5,411,405 FAILED reads / 0.0 fps  — never recovered

The contender's handle worked; the HOLDER's stream ended, permanently, and the
device stayed unopenable by either backend across five process restarts until
it reset itself ~50 minutes later. So ``require_frame`` protects the CONTENDER
and nothing here protects the HOLDER: the damage is done by the open itself,
before any verdict. The only defence is not to open a device someone in this
process already has — see ``_producer_camera_ownership`` in
skills/self_diagnostic.py, which is the fence, and the tests in
tests/skills/test_self_diagnostic_producer_ownership.py. Do not re-derive
"safe to open" from the cross-process numbers above.

Third measured quirk: MSMF fails ~3-7% of back-to-back reopens of an IDLE
camera with "backend is generally available but can't be used to capture by
index". It fails FAST (~40 ms, versus a successful open's ~400-490 ms) and one
retry always recovered it: 0 hard failures in 120 attempts across both
webcams. Hence ``retries``. DirectShow opened 60/60 first try on both, so this
is a real cost of MSMF — just a small and fully recoverable one.
"""
from __future__ import annotations

import ctypes
import os
import threading
import time

__all__ = [
    "MSMF_ENUM_TTL_SEC",
    "backend_name",
    "configured_backend",
    "msmf_device_names",
    "msmf_index_for_name",
    "msmf_index_for_dshow_index",
    "resolve_capture_target",
    "default_retries",
    "open_camera",
]

# The Media Foundation enumeration is measured free — +0.000 threads, +0.000
# handles, 1.2 ms median (see module docstring) — so this TTL is only a cache,
# NOT a ration on a leak the way the DirectShow gate must be. Losing it would
# cost milliseconds, not correctness.
MSMF_ENUM_TTL_SEC = 10.0

_enum_lock = threading.Lock()
# [names, symlinks, stamped_at]
_enum_cache: list = [None, None, 0.0]


# ── Media Foundation device enumeration (ctypes; no third-party dependency) ──
#
# OpenCV's MSMF backend (cap_msmf.cpp) opens a camera by subscripting exactly
# this array: MFEnumDeviceSources() filtered to the VIDCAP source type. So the
# position in THIS list is the integer cv2.VideoCapture(idx, CAP_MSMF) means.

class _GUID(ctypes.Structure):
    _fields_ = [("d1", ctypes.c_uint32), ("d2", ctypes.c_uint16),
                ("d3", ctypes.c_uint16), ("d4", ctypes.c_ubyte * 8)]

    def __init__(self, d1, d2, d3, d4):
        super().__init__()
        self.d1, self.d2, self.d3 = d1, d2, d3
        self.d4 = (ctypes.c_ubyte * 8)(*d4)


_SRC_TYPE = _GUID(0xC60AC5FE, 0x252A, 0x478F,
                  (0xA0, 0xEF, 0xBC, 0x8F, 0xA5, 0xF7, 0xCA, 0xD3))
_VIDCAP = _GUID(0x8AC3587A, 0x4AE7, 0x42D8,
                (0x99, 0xE0, 0x0A, 0x60, 0x13, 0xEE, 0xF9, 0x0F))
_FRIENDLY = _GUID(0x60D0E559, 0x52F8, 0x4FA2,
                  (0xBB, 0xCE, 0xAC, 0xDB, 0x34, 0xA8, 0xEC, 0x01))
_SYMLINK = _GUID(0x58F0AAD8, 0x22BF, 0x4F8A,
                 (0xBB, 0x3D, 0xD2, 0xC4, 0x97, 0x8C, 0x6E, 0x2F))
_MF_VERSION = (2 << 16) | 0x0070

# IMFAttributes vtable slots used below (IUnknown occupies 0-2).
_SLOT_RELEASE = 2
_SLOT_GET_ALLOCATED_STRING = 13
_SLOT_SET_GUID = 24


def _vcall(pobj, slot, restype, argtypes, *args):
    vtbl = ctypes.cast(pobj, ctypes.POINTER(ctypes.c_void_p))[0]
    fn = ctypes.cast(vtbl, ctypes.POINTER(ctypes.c_void_p))[slot]
    return ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *argtypes)(fn)(pobj, *args)


def _raw_msmf_devices() -> "tuple[list[str], list[str]] | None":
    """(friendly names, symbolic links) in Media Foundation index order, or
    None when Media Foundation is unavailable / the enumeration fails.

    NEVER raises: every caller treats None as "cannot translate" and falls back
    to the configured index, exactly as the DirectShow resolver does."""
    if os.name != "nt":
        return None
    try:
        ole32 = ctypes.WinDLL("ole32")
        mfplat = ctypes.WinDLL("mfplat")
        mf = ctypes.WinDLL("mf")
    except Exception:
        return None
    attrs = ctypes.c_void_p()
    ppdev = ctypes.POINTER(ctypes.c_void_p)()
    try:
        hr = mfplat.MFStartup(ctypes.c_uint32(_MF_VERSION), ctypes.c_uint32(1))
        if hr not in (0, 1):
            return None
        if mfplat.MFCreateAttributes(ctypes.byref(attrs), ctypes.c_uint32(1)) != 0:
            return None
        if _vcall(attrs, _SLOT_SET_GUID, ctypes.c_long,
                  (ctypes.POINTER(_GUID), ctypes.POINTER(_GUID)),
                  ctypes.byref(_SRC_TYPE), ctypes.byref(_VIDCAP)) != 0:
            return None
        count = ctypes.c_uint32()
        mf.MFEnumDeviceSources.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p)),
            ctypes.POINTER(ctypes.c_uint32)]
        if mf.MFEnumDeviceSources(attrs, ctypes.byref(ppdev),
                                  ctypes.byref(count)) != 0:
            return None

        def _string(dev, key):
            p = ctypes.c_wchar_p()
            ln = ctypes.c_uint32()
            if _vcall(dev, _SLOT_GET_ALLOCATED_STRING, ctypes.c_long,
                      (ctypes.POINTER(_GUID),
                       ctypes.POINTER(ctypes.c_wchar_p),
                       ctypes.POINTER(ctypes.c_uint32)),
                      ctypes.byref(key), ctypes.byref(p),
                      ctypes.byref(ln)) != 0:
                return ""
            val = p.value or ""
            ole32.CoTaskMemFree(ctypes.cast(p, ctypes.c_void_p))
            return val

        names, links = [], []
        for i in range(count.value):
            dev = ppdev[i]
            names.append(_string(dev, _FRIENDLY))
            links.append(_string(dev, _SYMLINK))
            # Release each IMFActivate we were handed. Skipping this is what
            # would turn a free enumeration into the very leak this module
            # exists to avoid.
            _vcall(dev, _SLOT_RELEASE, ctypes.c_ulong, ())
        ole32.CoTaskMemFree(ctypes.cast(ppdev, ctypes.c_void_p))
        ppdev = None
        return names, links
    except Exception:
        return None
    finally:
        try:
            if attrs:
                _vcall(attrs, _SLOT_RELEASE, ctypes.c_ulong, ())
        except Exception:
            pass


def msmf_device_names(now: "float | None" = None,
                      force: bool = False) -> "tuple[str, ...] | None":
    """Media Foundation video-capture device names, by MSMF index. TTL-cached.

    Returns None when the enumeration is unavailable — callers must then fall
    back to the configured index rather than guessing."""
    now = time.time() if now is None else now
    with _enum_lock:
        if (not force and _enum_cache[0] is not None
                and (now - _enum_cache[2]) < MSMF_ENUM_TTL_SEC):
            return tuple(_enum_cache[0])
    got = _raw_msmf_devices()
    if got is None:
        return None
    names, links = got
    with _enum_lock:
        _enum_cache[0] = list(names)
        _enum_cache[1] = list(links)
        _enum_cache[2] = now
    return tuple(names)


def msmf_index_for_name(substr: str,
                        now: "float | None" = None) -> "int | None":
    """MSMF index of the first device whose friendly name contains ``substr``
    (case-insensitive), or None. NEVER raises."""
    needle = (substr or "").strip().lower()
    if not needle:
        return None
    names = msmf_device_names(now=now)
    if not names:
        return None
    for i, n in enumerate(names):
        if needle in (n or "").lower():
            return i
    return None


def msmf_index_for_dshow_index(dshow_index: int,
                               dshow_names: "list[str] | tuple | None",
                               now: "float | None" = None) -> "int | None":
    """Translate a DIRECTSHOW index into the MSMF index for the same physical
    device, given the DirectShow name list the caller already holds.

    ``dshow_names`` is passed IN on purpose. The DirectShow enumeration is the
    leaky primitive the monolith gates behind a device fingerprint; if this
    module called it, there would be two gates to keep in step, and a rule
    fixed in one copy while the other rots is this codebase's most expensive
    bug shape. Pass None when you do not have the list — translation then
    fails closed (None) rather than guessing.

    Returns None when the device has no Media Foundation presence at all (OBS
    Virtual Camera is exactly this case on this rig), which is a real answer:
    that device can only be opened with DirectShow."""
    try:
        idx = int(dshow_index)
    except Exception:
        return None
    if not dshow_names or idx < 0 or idx >= len(dshow_names):
        return None
    name = (dshow_names[idx] or "").strip()
    if not name:
        return None
    names = msmf_device_names(now=now)
    if not names:
        return None
    low = name.lower()
    # EXACT NAME, MATCHED BY OCCURRENCE. Two identical webcams on one machine
    # produce two identical friendly names, and "the first device called X" is
    # then a coin flip that silently picks the wrong camera. Both enumerations
    # list devices in a stable per-name order, so match the Nth "USB 2.0
    # Camera" in the DirectShow list to the Nth in the Media Foundation list.
    # With one device of a given name — the ordinary case, and this rig — this
    # is identical to a plain first-match.
    exact = [i for i, n in enumerate(names)
             if (n or "").strip().lower() == low]
    if exact:
        nth = sum(1 for j in range(idx)
                  if (dshow_names[j] or "").strip().lower() == low)
        if nth < len(exact):
            return exact[nth]
        # More devices of this name on DirectShow than Media Foundation knows
        # about: we cannot say which one this is, so do not guess.
        return None
    # Friendly names are usually byte-identical across the two enumerations,
    # but fall back to a containment match rather than failing on a suffix.
    #
    # ONLY WHEN IT IS UNAMBIGUOUS. Two devices called "USB Camera" and
    # "USB Camera 2" both contain each other's stem, and picking the first
    # would hand the caller the WRONG PHYSICAL CAMERA — successfully, with
    # nothing to report, which is the exact failure this module exists to
    # prevent and is strictly worse than not translating at all. More than one
    # candidate means we do not know, so say so.
    hits = [i for i, n in enumerate(names)
            if (n or "").strip().lower()
            and ((n or "").strip().lower() in low
                 or low in (n or "").strip().lower())]
    return hits[0] if len(hits) == 1 else None


# ── backend selection ───────────────────────────────────────────────────────

def configured_backend() -> str:
    """'msmf' or 'dshow'. Config value, overridable per-run by JARVIS_CAMERA_BACKEND.

    Kept as a STRING rather than a cv2 constant so this is readable and
    assertable without cv2 installed (CI has no cv2 and no cameras)."""
    env = (os.environ.get("JARVIS_CAMERA_BACKEND") or "").strip().lower()
    if env in ("msmf", "dshow"):
        return env
    try:
        from core.config import CAMERA_BACKEND as _cfg
        val = (str(_cfg) or "").strip().lower()
        if val in ("msmf", "dshow"):
            return val
    except Exception:
        pass
    return "msmf"


def backend_name(api) -> str:
    """'dshow' / 'msmf' / 'any' for a cv2 CAP_* constant, for logging."""
    return {700: "dshow", 1400: "msmf", 0: "any"}.get(int(api or 0), str(api))


def resolve_capture_target(index, *, name: "str | None" = None,
                           dshow_names: "list[str] | tuple | None" = None,
                           backend: "str | None" = None,
                           now: "float | None" = None) -> "tuple[int, str, str]":
    """Decide (index_to_open, backend_to_use, why) for a configured camera.

    ``index`` is a DIRECTSHOW index — that is what data/user_settings.json
    CAMERAS holds and what _dshow_name_to_index() produces, and this function
    does not change that contract. ``name`` is the CAMERAS 'name' substring
    when there is one; matching on it needs no DirectShow enumeration at all,
    which is the only route to a true zero for the leaky call.

    Falls back to (index, 'dshow') whenever translation cannot be done
    honestly, so an untranslatable device still opens — on the leaky backend,
    which is strictly better than opening the WRONG camera successfully."""
    want = (backend or configured_backend()).lower()
    if want != "msmf":
        return int(index), "dshow", "backend pinned to dshow"
    if name:
        m = msmf_index_for_name(name, now=now)
        if m is not None:
            return m, "msmf", "matched CAMERAS name %r at msmf index %d" % (name, m)
    m = msmf_index_for_dshow_index(index, dshow_names, now=now)
    if m is not None:
        return m, "msmf", "translated dshow index %s -> msmf index %d" % (index, m)
    if msmf_device_names(now=now) is None:
        return int(index), "dshow", "media foundation enumeration unavailable"
    return int(index), "dshow", (
        "no media-foundation device matches dshow index %s "
        "(device is DirectShow-only)" % (index,))


# ── the shared opener ───────────────────────────────────────────────────────

def default_retries(backend: str) -> int:
    """How many RETRIES a failed open earns on ``backend``.

    MEASURED, not guessed (2026-09-05, 60 back-to-back opens per camera per
    backend, both webcams, device idle):

        CAP_MSMF   opened 93.3-96.7% first try; every single failure recovered
                   on one retry (0 hard failures in 120 attempts). A failed
                   MSMF open returns in ~40 ms and costs 0 handles.
        CAP_DSHOW  opened 60/60 first try on both cameras — it has no transient
                   failure to retry AWAY. And a DirectShow open that fails
                   still costs +103 handles, so a retry there is pure leak with
                   nothing to buy.

    So the retry is an MSMF-only remedy. Handing it to DirectShow would double
    the leak on exactly the path that leaks most: a dead or sick index."""
    return 1 if backend == "msmf" else 0


def open_camera(index, *, backend: str = "msmf", width=None, height=None,
                buffersize: bool = True, require_frame: float = 0.0,
                retries: "int | None" = None, retry_sleep: float = 0.25,
                cv2_mod=None, log=None, release_hook=None):
    """Open ``index`` on ``backend`` and hand back an opened capture, or None.

    This is the ONE place that knows the two MSMF hazards measured on this rig:

    * a ~3-7% transient open failure on back-to-back reopens, which fails in
      ~40 ms and always recovered on one retry (0/120 hard failures) — hence
      ``retries``, which defaults per BACKEND (see :func:`default_retries`:
      DirectShow gets none, because it has no transient failure and its failed
      opens still cost +103 handles each);
    * a BUSY device reporting isOpened() True, accepting set(), reading back
      the requested resolution, and then producing no frames at all — hence
      ``require_frame``, a seconds budget within which a real frame must
      arrive or the handle is released and None returned.

    ``require_frame`` is a BUDGET, not a single read, because a healthy camera
    can legitimately return False for its first reads while it warms up — the
    same behaviour _probe_camera_index() already retries around, which it
    documents as ~2 s for a since-retired Logi C270. On the two webcams
    actually attached today the first frame arrived 0.30-0.42 s after the
    resolution was set, in every cycle of a 25-cycle bench on each, so callers
    sizing this budget should treat a second as generous and anything above two
    as buying margin for a camera that is no longer here.

    Does not acquire any lock and does not bound itself in wall-clock time:
    callers already own _camera_io_lock and _open_capture_bounded, and layering
    a second timeout under theirs would only confuse the diagnosis. NEVER
    raises — returns None instead.

    ``release_hook`` REPLACES the plain ``cap.release()`` used on the failure
    paths (a refused open, a retry, a frameless handle). The monolith passes
    _release_on_current_camera_lock, and that matters more than it looks: this
    function runs on a throwaway open worker that may be ABANDONED and have its
    camera I/O lock RETIRED under it. A bare release from such a worker happens
    outside the lock the live threads now use, overlapping their camera I/O —
    which is the DirectShow heap corruption (0xc0000374) the whole locking
    scheme exists to prevent. The hook re-enters through the CURRENT lock
    object, so a late release is serialised after all."""
    if cv2_mod is None:
        try:
            import cv2 as cv2_mod  # type: ignore
        except Exception:
            return None
    api = cv2_mod.CAP_MSMF if backend == "msmf" else cv2_mod.CAP_DSHOW
    if retries is None:
        retries = default_retries(backend)

    def _note(msg):
        if log:
            try:
                log(msg)
            except Exception:
                pass

    attempts = max(1, int(retries) + 1)
    for attempt in range(attempts):
        cap = None
        try:
            cap = cv2_mod.VideoCapture(int(index), api)
            if not cap.isOpened():
                _note("[camera] %s index %s did not open (attempt %d/%d)"
                      % (backend, index, attempt + 1, attempts))
                _safe_release(cap, release_hook)
                if attempt + 1 < attempts:
                    time.sleep(max(0.0, retry_sleep))
                    continue
                return None
            if width and height:
                try:
                    cap.set(cv2_mod.CAP_PROP_FRAME_WIDTH, int(width))
                    cap.set(cv2_mod.CAP_PROP_FRAME_HEIGHT, int(height))
                except Exception:
                    pass
            if buffersize:
                # Several drivers reject this; it is an optimisation, not a
                # requirement, and DirectShow silently ignores it.
                try:
                    cap.set(cv2_mod.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass
            if require_frame and require_frame > 0:
                if not _wait_for_frame(cap, float(require_frame)):
                    _note("[camera] %s index %s opened but produced no frame "
                          "in %.1fs — treating as unavailable (this is what a "
                          "device held by another process looks like on MSMF)"
                          % (backend, index, require_frame))
                    _safe_release(cap, release_hook)
                    return None
            return cap
        except Exception:
            _safe_release(cap, release_hook)
            if attempt + 1 < attempts:
                time.sleep(max(0.0, retry_sleep))
                continue
            return None
    return None


def _wait_for_frame(cap, budget_s: float) -> bool:
    """True as soon as a non-empty frame arrives within ``budget_s``."""
    deadline = time.monotonic() + max(0.0, budget_s)
    while True:
        try:
            ret, frame = cap.read()
        except Exception:
            return False
        if ret and frame is not None and getattr(frame, "size", 0):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)


def _safe_release(cap, hook=None) -> None:
    """Release ``cap``, through ``hook`` when the caller supplied one.

    The hook is how a caller that owns a lock discipline keeps it: see the
    ``release_hook`` note in :func:`open_camera`. A hook that raises must not
    turn a failed open into an exception, so it is guarded like the release."""
    if cap is None:
        return
    try:
        if hook is not None:
            hook(cap)
            return
    except Exception:
        return
    try:
        cap.release()
    except Exception:
        pass
