#!/usr/bin/env python3
"""Hard memory ceiling for a JARVIS test run — the BOX must survive a leak.

THE INCIDENT (2026-08-20, 05:04)
================================
A test run with a runaway allocation committed ~144 GB of virtual memory on a
48 GB / 67 GB-commit-limit box. Windows did not kill the process; it paged the
machine to death and the kernel BUGCHECKED (0x0000000A). Nothing in the repo
stopped it, because nothing in the repo had ever *bounded* a test run.

This module is that bound. It is applied to the CURRENT process before a single
test is imported, so the ceiling is inherited by everything the run spawns:

  * Windows — a Job Object with JOB_OBJECT_LIMIT_PROCESS_MEMORY (per process)
    + JOB_OBJECT_LIMIT_JOB_MEMORY (the whole job, children included)
    + JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE (no child outlives the runner). Past
    the limit the kernel FAILS the commit: CPython raises MemoryError, or the
    process dies with STATUS_COMMITMENT_LIMIT (0xC000012D). Either way the
    machine keeps breathing.
  * POSIX — resource.setrlimit(RLIMIT_AS) to the same number; an over-cap
    allocation raises MemoryError.

Same family as the neighbouring ``_redirect_settings_to_throwaway()`` /
``_redirect_data_dir_to_throwaway()`` guards in tools/run_tests.py and
tools/run_tests_ci_sim.py: a test run must not be able to damage the box it
runs on. Those two protect the owner's live ``data/``; this one protects the
owner's RAM.

CONTRACT
--------
* NEVER raises, and never fails a run it cannot protect. If the OS refuses
  (missing privilege, an outer job that forbids nesting, an unsupported
  platform, a ctypes surface that isn't there) it prints a LOUD warning and
  returns False, and the run continues uncapped.
* Emits exactly ONE line, so any log makes it self-evident whether the run was
  capped and by what.
* Idempotent: the second and later calls are no-ops returning the first result.

CAP
---
``JARVIS_TEST_MEM_CAP_GB`` overrides; default is ``DEFAULT_CAP_GB`` (8 GB).
Rationale for 8: MEASURED 2026-08-20 — a full tools/run_tests_ci_sim.py run
(14,640 tests, 0 failed / 0 errored) peaked at 1,165 MB job-wide, parent plus
gate subprocesses; the heaviest single module (tests/test_actions_sec2.py) is
~1 GB on its own. 8 GB is therefore ~7x the real peak — a green run can never
trip it — while being ~1/6 of this box's 48 GB of RAM and ~1/8 of its 67 GB
commit limit, so a runaway is stopped long before the pager starts thrashing. On the ubuntu-latest
CI runner the same number is comfortably above the suite's address-space need
(note RLIMIT_AS bounds *reserved address space*, which runs higher than RSS).

``JARVIS_TEST_MEM_CAP_GB=0`` (also ``off`` / ``none`` / ``unlimited``) is the
documented, explicit escape hatch for the rare case that legitimately needs
more than the ceiling — it is deliberately loud in the log. Anything
unparseable or negative falls back to the default (fail *closed*: protected).

WINDOWS NESTED-JOB NOTE
-----------------------
Windows 8 / Server 2012 and later support nested jobs: a process that is
already in a job (a CI agent, a container, VS Code's job, or the scratchpad
``capped_run.py`` harness) can still be assigned to a new child job, and the
effective limit is the INTERSECTION of the chain — the most restrictive wins,
so nesting can only ever make the run safer, never looser. On older Windows, or
when an outer job was created with a flag that forbids it,
AssignProcessToJobObject fails with ERROR_ACCESS_DENIED (5); that is the warn
path, not an error.
"""
from __future__ import annotations

import math
import os
import sys

# Env override, and the default ceiling (see the module docstring for why 8).
CAP_ENV_VAR = "JARVIS_TEST_MEM_CAP_GB"
DEFAULT_CAP_GB = 8.0

# Explicit, documented escape hatch: switch the ceiling OFF entirely.
_OFF_WORDS = frozenset({"0", "off", "none", "no", "false",
                        "disable", "disabled", "unlimited"})

# Sanity clamp on the EFFECTIVE ceiling, in GB.
#   floor - below ~128 MB the interpreter cannot even finish importing, so a
#           fat-fingered value would "break a run it cannot protect".
#   roof  - MEASURED 2026-08-20: ctypes SILENTLY TRUNCATES an over-64-bit int
#           assigned to a c_size_t field (int(1e12 * 2**30) stored as
#           3830667724846006272 — mod 2**64, no exception). A big enough cap
#           could therefore WRAP to a tiny one and kill the run it was meant to
#           protect. 1 TB is astronomically above any real suite and nowhere
#           near the wrap point.
_MIN_CAP_GB = 0.125
_MAX_CAP_GB = 1024.0

_TAG = "[mem-guard]"

# Result of the first apply_memory_ceiling() call: (applied, message). Cached so
# repeated calls (run_coverage importing a runner, a runner calling main twice
# in-process, a test) neither re-apply nor re-print. See _reset_for_tests().
_state: tuple[bool, str] | None = None

# The Job Object handle is parked here FOR THE PROCESS LIFETIME on purpose: the
# job carries KILL_ON_JOB_CLOSE, so if this were a local that got closed/GC'd,
# closing the last handle would terminate every process in the job — including
# this one. Module-global == closed only at process exit, which is exactly when
# we *want* any surviving child to be reaped.
_job_handle: int | None = None


def _fmt_gb(gb: float) -> str:
    """8.0 -> '8', 0.5 -> '0.5' (log lines should read like a human wrote them)."""
    return str(int(gb)) if float(gb).is_integer() else f"{gb:g}"


def resolve_cap_gb(env: dict | None = None) -> float:
    """Return the ceiling in GB: 0.0 means OFF (explicitly disabled).

    Precedence: ``JARVIS_TEST_MEM_CAP_GB`` > ``DEFAULT_CAP_GB``. An empty /
    whitespace-only value is treated as unset (not as "off") so an env var that
    got exported empty by a wrapper script still leaves the run protected.
    Garbage or a negative number also falls back to the default — the failure
    mode of this knob must be "still capped", never "silently uncapped".
    """
    raw = (env if env is not None else os.environ).get(CAP_ENV_VAR)
    if raw is None or not str(raw).strip():
        return DEFAULT_CAP_GB
    text = str(raw).strip()
    if text.lower() in _OFF_WORDS:
        return 0.0
    try:
        val = float(text)
    except (TypeError, ValueError):
        return DEFAULT_CAP_GB
    # nan/inf parse fine as floats and would blow up the byte conversion later
    # (int(nan) raises ValueError) — that would make the guard RAISE, which its
    # contract forbids. Treat them like any other garbage.
    if not math.isfinite(val) or val <= 0:
        # 0 is handled above via _OFF_WORDS; a negative number is nonsense.
        return DEFAULT_CAP_GB
    return val


def _apply_windows_job_object(limit_bytes: int) -> str:
    """Cap THIS process (and its descendants) with a Job Object. Raises OSError
    on any failure; returns the mechanism label on success."""
    global _job_handle
    import ctypes
    import ctypes.wintypes as wintypes  # Windows-only: imported inside the branch

    JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x00000100
    JOB_OBJECT_LIMIT_JOB_MEMORY = 0x00000200
    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
    JobObjectExtendedLimitInformation = 9

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [("ReadOperationCount", ctypes.c_ulonglong),
                    ("WriteOperationCount", ctypes.c_ulonglong),
                    ("OtherOperationCount", ctypes.c_ulonglong),
                    ("ReadTransferCount", ctypes.c_ulonglong),
                    ("WriteTransferCount", ctypes.c_ulonglong),
                    ("OtherTransferCount", ctypes.c_ulonglong)]

    class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong),
                    ("PerJobUserTimeLimit", ctypes.c_longlong),
                    ("LimitFlags", wintypes.DWORD),
                    ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t),
                    ("ActiveProcessLimit", wintypes.DWORD),
                    # ULONG_PTR — pointer-sized, NOT DWORD. Getting this wrong
                    # misaligns every field after it on 64-bit.
                    ("Affinity", ctypes.c_size_t),
                    ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
                    ("IoInfo", IO_COUNTERS),
                    ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t)]

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    # Declare the prototypes. Default ctypes restype is c_int (32-bit signed) —
    # on 64-bit Windows that TRUNCATES a HANDLE, which fails intermittently and
    # only on machines whose handle values happen to go high.
    k32.CreateJobObjectW.restype = wintypes.HANDLE
    k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    k32.SetInformationJobObject.restype = wintypes.BOOL
    k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int,
                                            ctypes.c_void_p, wintypes.DWORD]
    k32.AssignProcessToJobObject.restype = wintypes.BOOL
    k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.GetCurrentProcess.argtypes = []

    job = k32.CreateJobObjectW(None, None)
    if not job:
        raise ctypes.WinError(ctypes.get_last_error())

    info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    info.ProcessMemoryLimit = limit_bytes
    info.JobMemoryLimit = limit_bytes
    info.BasicLimitInformation.LimitFlags = (
        JOB_OBJECT_LIMIT_PROCESS_MEMORY     # any ONE process over the line dies
        | JOB_OBJECT_LIMIT_JOB_MEMORY       # ...and so does the job in total
        | JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE)  # no child outlives the runner
    if not k32.SetInformationJobObject(job, JobObjectExtendedLimitInformation,
                                       ctypes.byref(info), ctypes.sizeof(info)):
        raise ctypes.WinError(ctypes.get_last_error())

    # GetCurrentProcess() is a pseudo-handle with full access, so it satisfies
    # AssignProcessToJobObject's PROCESS_SET_QUOTA | PROCESS_TERMINATE need with
    # no OpenProcess round trip. See the nested-job note in the module docstring
    # for why this can fail on a box that is already inside a hostile job.
    if not k32.AssignProcessToJobObject(job, k32.GetCurrentProcess()):
        raise ctypes.WinError(ctypes.get_last_error())

    _job_handle = job  # never closed before exit — see the global's comment
    return "Job Object"


def _apply_posix_rlimit(limit_bytes: int) -> str:
    """Cap this process with RLIMIT_AS. Raises on failure; returns the label."""
    import resource  # POSIX-only: imported inside the branch

    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    target = limit_bytes
    if hard != resource.RLIM_INFINITY:
        # Never try to RAISE the hard limit (unprivileged processes can't, and
        # the attempt would just raise); clamp under whatever we inherited.
        target = min(target, hard)
    resource.setrlimit(resource.RLIMIT_AS, (target, hard))
    return "RLIMIT_AS"


def _apply_ceiling(limit_bytes: int) -> str:
    """Platform dispatch. Returns the mechanism label, or raises.

    THE MOCK SEAM: tests replace this one function instead of the ctypes /
    resource calls underneath it, so no test has to allocate anything.
    """
    if os.name == "nt":
        # Deliberately os.name, not sys.platform: run_tests_ci_sim.py rewrites
        # sys.platform to "linux" to simulate CI, and os.name is the honest
        # kernel identity. (The guard runs before that flip anyway.)
        return _apply_windows_job_object(limit_bytes)
    if os.name == "posix":
        return _apply_posix_rlimit(limit_bytes)
    raise RuntimeError(f"unsupported platform os.name={os.name!r}")


def apply_memory_ceiling(cap_gb: float | None = None, *,
                         env: dict | None = None,
                         quiet: bool = False) -> bool:
    """Apply the process memory ceiling. Returns True iff a ceiling is in force.

    Never raises. Prints exactly one line describing what happened (applied,
    explicitly disabled, or refused-by-OS). Idempotent — later calls return the
    first result without re-applying or re-printing.
    """
    global _state
    if _state is not None:
        return _state[0]

    if cap_gb is None:
        gb = resolve_cap_gb(env)
    else:
        try:
            gb = float(cap_gb)
        except (TypeError, ValueError):
            # A caller passing nonsense must not take the run down either.
            gb = resolve_cap_gb(env)
    if gb <= 0:
        msg = (f"{_TAG} DISABLED via {CAP_ENV_VAR} — this run has NO memory "
               f"ceiling (a leak can take the whole box down)")
        _state = (False, msg)
    else:
        try:
            # Clamp before converting to bytes — see _MIN_CAP_GB / _MAX_CAP_GB
            # (the roof exists because ctypes truncates an oversized c_size_t
            # SILENTLY). Inside the try so even an arithmetic surprise on a
            # hand-passed cap_gb takes the warn-and-continue path.
            gb = min(max(gb, _MIN_CAP_GB), _MAX_CAP_GB)
            mechanism = _apply_ceiling(int(gb * (1 << 30)))
        except Exception as exc:  # noqa: BLE001 - a guard must never fail a run
            msg = (f"{_TAG} WARNING: could NOT apply the {_fmt_gb(gb)} GB "
                   f"ceiling ({type(exc).__name__}: {exc}) — continuing "
                   f"UNCAPPED; a runaway allocation can exhaust this machine")
            _state = (False, msg)
        else:
            msg = f"{_TAG} ceiling {_fmt_gb(gb)} GB ({mechanism})"
            _state = (True, msg)

    if not quiet:
        print(_state[1], flush=True)
    return _state[0]


def _reset_for_tests() -> None:
    """Drop the idempotence latch. Tests only — never call this from a runner
    (re-applying would create a second, redundant job object)."""
    global _state, _job_handle
    _state = None
    _job_handle = None


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    sys.exit(0 if apply_memory_ceiling() else 1)
