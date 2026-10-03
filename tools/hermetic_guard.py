#!/usr/bin/env python3
"""Process-wide refusal of a TEST RUN reaching the real world around it: the
network (and the live local services), the owner's keyboard / mouse / windows,
and live-hardware probes.

THE FINDINGS (2026-09-30)
=========================
With a live JARVIS up on the owner's PC, an audit of the whole suite (an audit
hook on every socket, spawn and user32 call, attributed to the running test)
found tests that reached the real world THROUGH PRODUCTION CODE while mocking
something else:

* the web dashboard suites GET the live Ollama ``/api/ps`` (``core/gpu_usage``)
  on every status build, and ran ``ollama ps`` and ``nvidia-smi``;
* the monolith's startup preflight GET the live Ollama ``/api/tags``;
* two Apple Music tests resolved and fetched ``itunes.apple.com``;
* ``test_system_pulse`` ran the real ``nvidia-smi``;
* a streaming test called ``SetForegroundWindow`` on one of the owner's real
  windows.

Every one of them passed, and passed for the wrong reason: green on this box
because a live service or a GPU happened to answer, and different on the
GPU-less, service-less CI runner. The rule is the one the browser guard was
written for: **a unit test must never touch a real browser, camera, mic,
sensor, or network.** ``tools/browser_guard.py`` (browsers),
``tests/live_data_guard.py`` (the owner's live ``data/``) and
``tools/mem_guard.py`` (RAM) cover the others; this module is the network /
input / hardware-probe sibling.

THREE GUARDS, ONE CHOKEPOINT
============================
One ``sys.addaudithook`` hook. CPython raises the audit events this reads from
C, below every library: ``socket.connect`` / ``bind`` / ``sendto`` /
``sendmsg`` / ``getaddrinfo`` / ``gethostbyname`` / ``gethostbyaddr`` (so
requests, urllib3, http.client and asyncio all land here),
``subprocess.Popen`` / ``os.system`` / ``os.spawn`` / ``os.posix_spawn``, and
``ctypes.call_function`` (every foreign call - pyautogui, pygetwindow,
keyboard and pynput all reach user32 through ctypes on Windows). A hook cannot
be removed, displaced by a test's ``mock.patch`` or undone by a
``_reset_for_tests()``, which closes the "reports itself armed while a hole is
open" class that bit three times on 2026-08-20. Armed-ness is still answered
BY BEHAVIOUR: ``unarmed_guards()`` raises a synthetic event through
``sys.audit()`` for each guard and checks that the hook refuses it (no I/O).

pywin32 is C code whose calls are NOT audited, so its input functions
(``win32api.SetCursorPos`` / ``mouse_event`` / ``keybd_event`` / ``PostMessage``
/ ``SendMessage``, ``win32gui.PostMessage`` / ``SendMessage`` /
``SendMessageTimeout`` / ``SetForegroundWindow`` / ``BringWindowToTop``) are
wrapped the browser-guard way: a marked stub on the module attribute, which a
test's ``mock.patch`` replaces and then restores TO the stub. COM / WinRT
calls are not audited either, so the two real-world effects reached that way
(``_EFFECT_TARGETS``: pycaw's ``AudioUtilities.GetSpeakers``, the speakers'
mute / volume, and ``core.media_now_playing._default_transport``, the media
session's pause / skip) get the same stub, installed by an import hook as the
REAL module loads (2026-10-01).

WHAT IS REFUSED
---------------
``[network-guard]`` (raises ``NetworkGuardError``, a ``ConnectionRefusedError``
- so production code takes its "service down" path, exactly as on the CI
runner; a refused lookup raises ``socket.gaierror``):

* a connect (TCP or UDP) to any address that is not this machine;
* a connect to a LIVE local service on loopback - ``LIVE_SERVICE_PORTS``
  (Ollama 11434, the JARVIS web UI 8766, ComfyUI 8188, OBS websocket 4455, the
  MCP SSE bridge 8788, the clone voice server 8767);
* a bind to one of those ports (on Windows ``SO_REUSEADDR`` lets a second
  socket share - i.e. hijack - the live web UI's port);
* a name lookup that would leave the box (anything but ``localhost``, a
  numeric address or this machine's own name);
* a datagram (``sendto`` / ``sendmsg``) to a non-loopback address - the LAN
  discovery broadcasts.

Loopback on every other port passes, so test-local servers on ephemeral ports
keep working.

``[input-guard]`` (raises ``InputGuardError``, an ``OSError``): synthetic
keyboard / mouse input (``SendInput``, ``keybd_event``, ``mouse_event``) and
moving the real cursor (``SetCursorPos``), always; and, aimed at a window of
ANOTHER process (the owner's desktop - a test's own windows are its business),
stealing focus (``SetForegroundWindow``, ``BringWindowToTop``,
``SwitchToThisWindow``), moving / resizing / minimising / maximising
(``ShowWindow``, ``SetWindowPos``, ``MoveWindow``, ``SetWindowPlacement``,
``CloseWindow``), and a posted / sent message that types, clicks, closes or
sys-commands it (``PostMessage`` / ``SendMessage`` / ``SendMessageTimeout``
with a keyboard, mouse, WM_CLOSE / WM_QUIT, WM_SYSCOMMAND or WM_APPCOMMAND
message).

``[probe-guard]`` (raises ``ProbeGuardError``, a ``FileNotFoundError`` - "not
installed", the CI runner's own answer): launching a live-hardware or
live-service probe - ``PROBE_PROGRAMS`` (``nvidia-smi``, the ``ollama`` CLI,
``powercfg``, ``pnputil``, the audio-device switchers, ``shutdown``, the
network probes, the ``claude`` CLI ...) directly or through a shell, a
PowerShell command line that queries or changes devices
(``SHELL_PROBE_PATTERNS``), a python launch of one of the project's HUD
overlays (``hud/*.py``: a real window on the owner's desktop), and a desktop
file launcher - ``LAUNCHER_PROGRAMS`` (``xdg-open``, ``open``, ``explorer``
...), ``start`` / ``Start-Process`` / ``Invoke-Item`` in command position of a
shell command line (``SHELL_LAUNCHERS``), and ``os.startfile``: each opens a
file in its registered app on the owner's desktop.

REPORTING
---------
Every refusal is recorded with the test that made it (the running unittest
method on the calling stack; for a background thread, the test the MAIN thread
was running at that moment) and the production call site, and an atexit
summary names each offender, one line per distinct call site - the
browser guard's shape. Silence means nothing was refused.

OPT-IN AND ESCAPE HATCHES
-------------------------
A test that genuinely needs one of these opts in explicitly::

    with hermetic_guard.allow("network", reason="exercises a real socket"):
        ...

(also usable as a decorator; it covers every thread while it is open).
A human driving the real thing by hand sets ``JARVIS_ALLOW_REAL_NETWORK=1``,
``JARVIS_ALLOW_REAL_INPUT=1`` or ``JARVIS_ALLOW_HARDWARE_PROBES=1`` - each is
announced loudly in the banner. ``install(record_only=True)`` records without
refusing, for confirming a suspected offender.

CONTRACT
--------
Never raises out of ``install()``; idempotent; exactly one banner line; the
hook itself never raises anything but its own refusals.

WIRING
------
``tests/__init__.py`` (the chokepoint every entry path imports) and the three
``tools/`` runners, beside the browser guard.
"""
from __future__ import annotations

import atexit
import contextlib
import errno
import ipaddress
import os
import re
import socket
import sys
import threading
import typing

_TAGS = {"network": "[network-guard]", "input": "[input-guard]",
         "probe": "[probe-guard]"}
GUARDS = ("network", "input", "probe")

ENV_ESCAPES = {
    "network": "JARVIS_ALLOW_REAL_NETWORK",
    "input": "JARVIS_ALLOW_REAL_INPUT",
    "probe": "JARVIS_ALLOW_HARDWARE_PROBES",
}
_ALLOW_WORDS = frozenset({"1", "true", "yes", "on", "allow", "enable",
                          "enabled"})

# Live LOCAL services a test must never talk to, even on loopback. Names are
# for the report only.
LIVE_SERVICE_PORTS = {
    11434: "Ollama",
    8766: "the JARVIS web UI",
    8188: "ComfyUI",
    4455: "the OBS websocket",
    8788: "the MCP SSE bridge",
    # The clone voice server's default port (core/clone_voice_client.py): a
    # request there renders on the owner's GPU. Tests use a fake server on an
    # ephemeral port.
    8767: "the clone voice server",
}

# Programs whose launch IS a live-hardware / live-service / network probe (or
# changes live hardware state). Matched on the basename, without ".exe".
PROBE_PROGRAMS = frozenset({
    # GPU / sensors
    "nvidia-smi", "rocm-smi", "intel_gpu_top", "hwinfo64", "hwinfo32",
    "hwinfo", "smartctl", "lhm", "librehardwaremonitor",
    # the local model server's CLI talks to the LIVE daemon (`ollama ps`,
    # and `ollama stop <model>` would unload the owner's brain)
    "ollama",
    # devices, power, audio routing
    "powercfg", "pnputil", "devcon", "nircmd", "nircmdc", "soundvolumeview",
    "svcl", "endpointcontroller", "setvol", "shutdown", "bluetoothctl",
    # the network, via a child process
    "ping", "tracert", "traceroute", "pathping", "nslookup", "arp", "netsh",
    "getmac", "curl", "wget", "tailscale",
    # the cloud coding CLI the self-upgrade pipeline drives: a real call
    # bills the owner and edits files
    "claude",
})

# Desktop file launchers: each hands a file (or URL) to its registered app ON
# THE OWNER'S DESKTOP - an editor, an image viewer, Explorer, a browser tab.
# Found 2026-09-30: ReadChangelogTests ran `xdg-open CHANGELOG.md` under
# ci-sim's Linux simulation; on this box xdg-open is simply absent, so the
# launch failed quietly and the test passed. Matched on the basename, like
# PROBE_PROGRAMS. (URL launches are the browser guard's too; it refuses those
# first, above the audit hook.)
LAUNCHER_PROGRAMS = frozenset({
    "xdg-open", "open", "gnome-open", "kde-open", "kde-open5", "exo-open",
    "wslview", "explorer",
})
# ... and the shell built-ins / cmdlets that do the same, refused only in
# COMMAND position (the first word after -Command or /c, or after a ; & |),
# so a bare "start" or "open" ARGUMENT is not mistaken for one.
SHELL_LAUNCHERS = LAUNCHER_PROGRAMS | frozenset({
    "start", "start-process", "saps", "invoke-item", "ii",
})
_SHELL_SEPARATORS = frozenset({"&", "&&", "|", "||", ";"})

# The project's HUD overlays (hud/*.py) are separate Qt / tk processes that
# draw on the owner's desktop; a test that launches one puts a real window on
# his screen for as long as the test run lives (--parent-pid). Found
# 2026-09-30: skills/holographic_overlay's register() auto-launches the
# workshop HUD, so the skill smoke tests put it on screen twice per run.
_PYTHON_RE = re.compile(r"^(?:python|pythonw|py)(?:\d+(?:\.\d+)*)?w?$")

_SHELLS = frozenset({"powershell", "pwsh", "cmd", "bash", "sh", "wsl"})

# A shell command line that queries or changes live devices.
SHELL_PROBE_PATTERNS = tuple(re.compile(p, re.IGNORECASE) for p in (
    r"\b(?:Get|Enable|Disable|Restart)-PnpDevice\b",
    r"\bGet-PnpDeviceProperty\b",
    r"\b(?:Get-CimInstance|Get-WmiObject|gwmi|gcim)\b[^|;]*\b(?:Win32_(?:"
    r"VideoController|PnPEntity|PnPSignedDriver|SoundDevice|Battery|"
    r"PortableBattery|DiskDrive|USBControllerDevice|USBHub|Keyboard|"
    r"PointingDevice|DesktopMonitor|NetworkAdapter|PowerPlan)|MSAcpi_|"
    r"WmiMonitor)",
    r"\b(?:Get|Set)-AudioDevice\b",
    r"\b(?:Restart|Stop)-Computer\b",
    r"\bGet-NetAdapter\b",
    r"\bTest-(?:Connection|NetConnection)\b",
    r"\bInvoke-(?:WebRequest|RestMethod)\b",
))

_TOKEN_RE = re.compile(r'"[^"]*"|\'[^\']*\'|\S+')
_CMDLINE_SWITCHES = frozenset({
    "-c", "--command", "-command", "/c", "/k", "-e", "-ec",
    "-encodedcommand",
})

# user32 functions that inject input or move the cursor: global, always refused.
_INPUT_FUNCTIONS = ("SendInput", "keybd_event", "mouse_event", "SetCursorPos")
# ... that steal focus or move / resize / minimise / maximise a window:
# refused when the window (argument 0) belongs to ANOTHER process - the
# owner's. (Found 2026-09-30: the streaming auto-play tests look up a real
# browser window by title, then activate() and maximise it.)
_WINDOW_FUNCTIONS = ("SetForegroundWindow", "BringWindowToTop",
                     "SwitchToThisWindow", "ShowWindow", "ShowWindowAsync",
                     "SetWindowPos", "MoveWindow", "SetWindowPlacement",
                     "CloseWindow")
# ... and the message senders, refused for these messages to another
# process's window (pygetwindow's close() is PostMessage(WM_CLOSE)).
_MESSAGE_FUNCTIONS = ("PostMessageW", "PostMessageA", "SendMessageW",
                      "SendMessageA", "SendMessageTimeoutW",
                      "SendMessageTimeoutA", "SendNotifyMessageW",
                      "SendNotifyMessageA")
_BLOCKED_MESSAGES = {
    0x0010: "WM_CLOSE", 0x0012: "WM_QUIT", 0x0100: "WM_KEYDOWN",
    0x0101: "WM_KEYUP", 0x0102: "WM_CHAR", 0x0103: "WM_DEADCHAR",
    0x0104: "WM_SYSKEYDOWN", 0x0105: "WM_SYSKEYUP", 0x0106: "WM_SYSCHAR",
    0x0112: "WM_SYSCOMMAND", 0x0319: "WM_APPCOMMAND",
}
_BLOCKED_MESSAGES.update({m: "WM_MOUSE" for m in range(0x0200, 0x020F)})

# Real-world effects with NO audited call underneath (2026-10-01, actions-a
# review): COM and WinRT method calls raise no audit event, so the endpoint
# volume behind volume_mute / volume_unmute / set_volume (pycaw) and the
# media-session transport behind pause_music / resume_music / next_song /
# previous_song would really mute the owner's speakers or pause / skip his
# media from an unpinned test. (The media KEYS those actions used to press go
# through keybd_event and were refused; their replacements were not.) Each
# REAL entry point gets a marked stub under the input guard, the pywin32 way;
# a test that fakes the module in sys.modules, or pins the function, never
# reaches it. Wrapped as each module is IMPORTED (an import hook), never
# imported here: importing pycaw would CoInitialize the collecting thread and
# cost every run the comtypes import. (module, class or None, attribute)
_EFFECT_TARGETS = (
    ("core.media_now_playing", None, "_default_transport"),
    ("pycaw.utils", "AudioUtilities", "GetSpeakers"),
)
_EFFECT_MODULES = frozenset(t[0] for t in _EFFECT_TARGETS)

# pywin32 attributes wrapped (their calls are not audited).
_PYWIN32_TARGETS = (
    ("win32api", ("SetCursorPos", "mouse_event", "keybd_event",
                  "PostMessage", "SendMessage")),
    ("win32gui", ("PostMessage", "SendMessage", "SendMessageTimeout",
                  "SetForegroundWindow", "BringWindowToTop", "ShowWindow",
                  "SetWindowPos", "MoveWindow", "SetWindowPlacement",
                  "CloseWindow")),
)

_GUARD_MARK = "_jarvis_hermetic_guard"
_THIS_FILE = os.path.normcase(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ROOT_N = os.path.normcase(_PROJECT_ROOT) + os.sep
# Frames that are never "the call site": this module, its guard siblings and
# the process-wide Popen shim - each wraps the real call, none of them is it.
_SIBLING_FILES = frozenset(os.path.normcase(os.path.join(_PROJECT_ROOT, *p))
                           for p in (("tools", "hermetic_guard.py"),
                                     ("tools", "browser_guard.py"),
                                     ("tests", "live_data_guard.py"),
                                     ("core", "no_window_subprocess.py")))
_STDLIB_N = os.path.normcase(os.path.dirname(os.__file__)) + os.sep


class NetworkGuardError(ConnectionRefusedError):
    """A test tried to reach the network or a live local service."""


class InputGuardError(OSError):
    """A test tried to inject real input or steal focus."""


class ProbeGuardError(FileNotFoundError):
    """A test tried to launch a live-hardware / live-service probe."""


class Refusal(typing.NamedTuple):
    guard: str        # "network" | "input" | "probe"
    api: str          # the audited operation, e.g. "socket.connect"
    target: str       # what it was aimed at
    test_id: str      # the unittest method (or the one running at the time)
    call_site: str    # the production frame that made the call


# ─── state ────────────────────────────────────────────────────────────────
_lock = threading.Lock()
_refusals: list[Refusal] = []
_installed = [False]
_hook_added = [False]
_banner: list[str] = [""]
_record_only = [False]
_disabled: set[str] = set()       # guards turned off by an env escape hatch
_opt_in = {g: 0 for g in GUARDS}  # allow() depth per guard
_probing = threading.local()      # set while unarmed_guards() probes
_busy = threading.local()         # re-entrancy latch inside the hook
_input_fn_ptrs: dict[int, str] = {}
_message_fn_ptrs: dict[int, str] = {}
_own_names: set[str] = set()
_atexit_registered = [False]
_effect_modules: dict[str, object] = {}  # name -> the REAL module wrapped
_finding = threading.local()      # re-entrancy latch inside the import hook
_EVENTS = frozenset({
    "socket.connect", "socket.bind", "socket.sendto", "socket.sendmsg",
    "socket.getaddrinfo", "socket.gethostbyname", "socket.gethostbyaddr",
    "subprocess.Popen", "os.system", "os.spawn", "os.posix_spawn",
    "os.startfile", "os.startfile/2",
    "ctypes.call_function",
})


def _env_allows(guard: str, env=None) -> bool:
    raw = (env if env is not None else os.environ).get(ENV_ESCAPES[guard])
    return raw is not None and str(raw).strip().lower() in _ALLOW_WORDS


# ─── pure verdicts (the testable core) ─────────────────────────────────────

def _host_text(host) -> str:
    if isinstance(host, (bytes, bytearray)):
        host = bytes(host).decode("ascii", "replace")
    return str(host).strip().strip("[]").lower()


def is_local_host(host) -> bool:
    """True when ``host`` never leaves this machine: loopback (any spelling),
    the unspecified address, ``localhost`` / ``*.localhost``, or this
    machine's own name. None / "" mean "local" to the socket API."""
    if host is None:
        return True
    h = _host_text(host)
    if h in ("", "localhost", "localhost.localdomain", "ip6-localhost",
             "ip6-loopback") or h.endswith(".localhost"):
        return True
    if "%" in h:                          # fe80::1%eth0 -> fe80::1
        h = h.split("%", 1)[0]
    try:
        ip = ipaddress.ip_address(h)
    except ValueError:
        return h in _own_names
    mapped = getattr(ip, "ipv4_mapped", None)
    if mapped is not None:
        ip = mapped
    return bool(ip.is_loopback or ip.is_unspecified)


def _is_numeric(host) -> bool:
    try:
        ipaddress.ip_address(_host_text(host).split("%", 1)[0])
        return True
    except ValueError:
        return False


def network_verdict(event: str, args: tuple) -> str | None:
    """Why this socket event must be refused, or None."""
    if event in ("socket.connect", "socket.sendto", "socket.sendmsg",
                 "socket.bind"):
        addr = args[1] if len(args) > 1 else None
        if not isinstance(addr, tuple) or not addr:
            return None                   # AF_UNIX path, or no address
        host = addr[0]
        port = addr[1] if len(addr) > 1 and isinstance(addr[1], int) else None
        if event == "socket.bind":
            if port in LIVE_SERVICE_PORTS:
                return (f"bind to port {port} ({LIVE_SERVICE_PORTS[port]}'s "
                        f"live port)")
            return None
        if not is_local_host(host):
            return f"{_host_text(host)}:{port} is not this machine"
        if event == "socket.connect" and port in LIVE_SERVICE_PORTS:
            return f"port {port} is {LIVE_SERVICE_PORTS[port]} (a live local service)"
        return None
    if event in ("socket.getaddrinfo", "socket.gethostbyname"):
        host = args[0] if args else None
        if is_local_host(host) or _is_numeric(host):
            return None
        return f"name lookup of {_host_text(host)!r} leaves this machine"
    if event == "socket.gethostbyaddr":
        host = args[0] if args else None
        if is_local_host(host):
            return None
        return f"reverse lookup of {_host_text(host)!r} leaves this machine"
    return None


def _as_text(item) -> str:
    try:
        item = os.fspath(item)
    except TypeError:
        pass
    if isinstance(item, (bytes, bytearray)):
        item = bytes(item).decode("utf-8", "replace")
    return item if isinstance(item, str) else ""


def _command_tokens(args) -> list[str]:
    """Tokens of a command: a string is split quote-aware (Windows audits the
    list2cmdline STRING); a sequence keeps its elements, and an element that
    follows a command-line switch (``-Command`` / ``/c`` / ``-c``) is split
    too - the shape of every shell-mediated launch."""
    if isinstance(args, (list, tuple)):
        elements = [_as_text(a) for a in args if a is not None]
        tokens = list(elements)
        prev = ""
        for item in elements:
            if prev in _CMDLINE_SWITCHES and item:
                tokens.extend(m.group(0) for m in _TOKEN_RE.finditer(item))
            prev = item.strip().strip('"').strip("'").lower()
        return tokens
    text = _as_text(args)
    tokens = [m.group(0) for m in _TOKEN_RE.finditer(text)]
    out = list(tokens)
    for i, tok in enumerate(tokens[:-1]):
        if tok.strip('"\'').lower() in _CMDLINE_SWITCHES:
            inner = tokens[i + 1].strip('"').strip("'")
            out.extend(m.group(0) for m in _TOKEN_RE.finditer(inner))
    return out


def _program(token: str) -> str:
    """'C:\\WINDOWS\\system32\\nvidia-smi.EXE' -> 'nvidia-smi'."""
    t = _as_text(token).strip().strip('"').strip("'").replace("\\", "/")
    name = t.rsplit("/", 1)[-1].strip().lower()
    return name[:-4] if name.endswith(".exe") else name


def _hud_script(tokens: list[str]) -> str:
    """The HUD overlay a python command line launches (``hud/<x>.py`` or
    ``-m hud.<x>``), else ""."""
    for i, tok in enumerate(tokens[1:], 1):
        t = _as_text(tok).strip().strip('"').strip("'").replace("\\", "/")
        if t == "-m" and i + 1 < len(tokens):
            mod = tokens[i + 1].strip('"\'')
            return mod if mod.startswith("hud.") else ""
        if t.endswith(".py"):
            parts = t.rsplit("/", 2)
            return parts[-1] if len(parts) >= 2 and parts[-2] == "hud" else ""
    return ""


def _shell_command_words(tokens: list[str]) -> list[str]:
    """The program names in COMMAND position of a shell command line: the
    first word after a ``-Command`` / ``/c`` switch, and the first after each
    ``;`` ``&`` ``|`` separator (a separate token, or glued to the end of the
    previous one)."""
    words: list[str] = []
    expect = False
    for tok in tokens[1:]:
        t = _as_text(tok).strip().strip('"').strip("'").strip()
        low = t.lower()
        if low in _CMDLINE_SWITCHES or low in _SHELL_SEPARATORS:
            expect = True
            continue
        if expect and t:
            # A quoted command string's first word ("Start-Process 'x.md'").
            words.append(_program(t.split()[0].rstrip(";&|")))
            expect = False
        if t and t[-1] in ";&|":
            expect = True
    return words


def probe_verdict(executable, args) -> str | None:
    """Why launching this command must be refused, or None."""
    tokens = _command_tokens(args)
    first = _program(tokens[0]) if tokens else ""
    exe = _program(executable) if executable else ""
    for name in (exe, first):
        if name in PROBE_PROGRAMS:
            return f"{name} is a live-hardware / live-service probe"
        if name in LAUNCHER_PROGRAMS:
            return (f"{name} is a desktop file launcher (it opens the file in "
                    f"its app on the owner's desktop)")
    if _PYTHON_RE.match(first) or _PYTHON_RE.match(exe):
        hud = _hud_script(tokens)
        if hud:
            return f"{hud} is a HUD overlay (a real window on the owner's desktop)"
    if first in _SHELLS or exe in _SHELLS:
        for tok in tokens[1:]:
            name = _program(tok)
            if name in PROBE_PROGRAMS:
                return f"{name} (through {first or exe}) is a live probe"
        for name in _shell_command_words(tokens):
            if name in SHELL_LAUNCHERS:
                return (f"{name} (through {first or exe}) is a desktop file "
                        f"launcher")
        text = " ".join(tokens)
        for rx in SHELL_PROBE_PATTERNS:
            m = rx.search(text)
            if m:
                return f"{m.group(0)!r} queries or changes live devices"
    return None


def _int_arg(value):
    try:
        return int(getattr(value, "value", value))
    except Exception:  # noqa: BLE001 - not a number: not a blocked message
        return None


_MESSAGE_BASES = ("PostMessage", "SendMessage", "SendMessageTimeout",
                  "SendNotifyMessage")
_get_window_pid: list = [None]


def _foreign_window(hwnd) -> bool:
    """True when window ``hwnd`` belongs to ANOTHER process (the owner's
    desktop); False for none (0 / NULL targets the caller's own queue) or one
    of this process's own windows. Unknown -> True (refuse). Never raises."""
    h = _int_arg(hwnd)
    if not h:
        return False
    try:
        import ctypes
        from ctypes import wintypes
        fn = _get_window_pid[0]
        if fn is None:
            fn = ctypes.WinDLL("user32").GetWindowThreadProcessId
            fn.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
            fn.restype = wintypes.DWORD
            _get_window_pid[0] = fn
        pid = wintypes.DWORD(0)
        fn(h, ctypes.byref(pid))
        return int(pid.value) != os.getpid()
    except Exception:  # noqa: BLE001 - cannot tell: treat it as the owner's
        return True


def input_verdict(function: str, args: tuple, foreign=None) -> str | None:
    """Why this user32 call must be refused, or None. ``function`` is the
    user32 export name (or the pywin32 attribute name); ``foreign(hwnd)``
    answers "is that window another process's?" (default: always yes)."""
    foreign = foreign if foreign is not None else (lambda _h: True)
    base = function
    if base[-1:] in ("W", "A") and base[:-1] in _MESSAGE_BASES:
        base = base[:-1]
    if function in _INPUT_FUNCTIONS:
        return f"{function} drives the real keyboard / mouse"
    hwnd = args[0] if args else None
    if base in _MESSAGE_BASES:
        msg = _int_arg(args[1]) if len(args) > 1 else None
        if msg in _BLOCKED_MESSAGES and foreign(hwnd):
            return (f"{function}({_BLOCKED_MESSAGES[msg]}) to another "
                    f"process's window")
        return None
    if function in _WINDOW_FUNCTIONS and foreign(hwnd):
        return f"{function} on another process's window (the owner's desktop)"
    return None


# ─── recording ─────────────────────────────────────────────────────────────

def _short(path: str) -> str:
    try:
        full = os.path.normcase(os.path.abspath(path))
        if full.startswith(_ROOT_N):
            return os.path.relpath(path, _PROJECT_ROOT).replace(os.sep, "/")
        return os.path.basename(path) or path
    except Exception:  # noqa: BLE001
        return str(path)


def _test_of(obj, name: str | None) -> str:
    """``module.Class.method`` when ``obj`` is a TestCase running ``name``
    (or any method, when ``name`` is None), else ""."""
    try:
        method = object.__getattribute__(obj, "_testMethodName")
    except Exception:  # noqa: BLE001 - not a TestCase (or a mock)
        return ""
    if not isinstance(method, str) or (name is not None and method != name):
        return ""
    cls = type(obj)
    return f"{cls.__module__}.{cls.__qualname__}.{method}"


def _stack_test(frame, exact: bool) -> str:
    depth = 0
    while frame is not None and depth < 200:
        depth += 1
        obj = frame.f_locals.get("self")
        if obj is not None:
            tid = _test_of(obj, frame.f_code.co_name if exact else None)
            if tid:
                return tid
        frame = frame.f_back
    return ""


def _caller() -> tuple[str, str]:
    """(test_id, call_site). The test is the unittest method on this stack;
    on a background thread (or in setUp) it is the test the MAIN thread is
    running right now. The call site is the nearest project frame outside the
    guards, else the nearest non-stdlib frame."""
    try:
        frame = sys._getframe(2)
    except Exception:  # noqa: BLE001
        return "", ""
    site = fallback = ""
    f = frame
    depth = 0
    while f is not None and depth < 200:
        depth += 1
        fn = f.f_code.co_filename or ""
        n = os.path.normcase(os.path.abspath(fn)) if fn else ""
        if n and n not in _SIBLING_FILES and n != _THIS_FILE:
            where = f"{_short(fn)}:{f.f_lineno} in {f.f_code.co_name}()"
            if n.startswith(_ROOT_N):
                site = where
                break
            if not fallback and not n.startswith(_STDLIB_N) \
                    and not fn.startswith("<"):
                fallback = where
        f = f.f_back
    test_id = _stack_test(frame, exact=True)
    if not test_id:
        try:
            main = sys._current_frames().get(threading.main_thread().ident)
        except Exception:  # noqa: BLE001
            main = None
        during = _stack_test(main, exact=False) if main is not None else ""
        if during:
            test_id = during + " (during)"
    return test_id, site or fallback


def _record(guard: str, api: str, target: str) -> str:
    """Log one refusal; returns the test it is attributed to ("" unknown)."""
    try:
        test_id, site = _caller()
        with _lock:
            _refusals.append(Refusal(guard, api, str(target)[:160], test_id,
                                     site))
        return test_id
    except Exception:  # noqa: BLE001 - recording must never fail the refusal
        return ""


def refusals(guard: str | None = None) -> tuple[Refusal, ...]:
    with _lock:
        return tuple(r for r in _refusals if guard is None or r.guard == guard)


def reset() -> None:
    """Forget the recorded refusals (the guards stay armed)."""
    with _lock:
        _refusals.clear()


# ─── the hook ──────────────────────────────────────────────────────────────

def _active(guard: str) -> bool:
    return (_installed[0] and guard not in _disabled
            and _opt_in[guard] <= 0)


def _refuse(guard: str, api: str, target: str, why: str):
    # unarmed_guards()'s synthetic probe is a diagnostic, not an offence: it
    # never reaches the ledger the next agent reads to find offenders.
    test_id = ""
    if not getattr(_probing, "on", False):
        test_id = _record(guard, api, target)
    if _record_only[0]:
        return                            # recorded; the real call proceeds
    msg = (f"{_TAGS[guard]} REFUSED {api} -> {target}"
           f"{' in ' + test_id if test_id else ''}: {why}. A unit test "
           f"must not reach the real world - fake the boundary in the test "
           f"(or opt in with tools.hermetic_guard.allow({guard!r})).")
    if guard == "network":
        if api in ("socket.getaddrinfo", "socket.gethostbyname",
                   "socket.gethostbyaddr"):
            raise socket.gaierror(getattr(socket, "EAI_NONAME", -2), msg)
        raise NetworkGuardError(errno.ECONNREFUSED, msg)
    if guard == "input":
        raise InputGuardError(errno.EACCES, msg)
    raise ProbeGuardError(errno.ENOENT, msg, target)


def _describe_addr(args) -> str:
    try:
        if len(args) > 1 and isinstance(args[1], tuple):
            return f"{_host_text(args[1][0])}:{args[1][1]}"
        return _host_text(args[0])
    except Exception:  # noqa: BLE001
        return "<address>"


_REFUSALS = (NetworkGuardError, InputGuardError, ProbeGuardError,
             socket.gaierror)


def _hook(event, args):
    if event not in _EVENTS:
        return
    if getattr(_busy, "on", False):
        return
    _busy.on = True
    try:
        _decide(event, args)
    except _REFUSALS:
        raise
    except Exception:  # noqa: BLE001 - a guard bug must never break the call
        pass           # (unarmed_guards() would then report the guard down)
    finally:
        _busy.on = False


def _decide(event, args) -> None:
    """The hook's body: raise a refusal, or return."""
    if event.startswith("socket."):
        if not _active("network"):
            return
        why = network_verdict(event, args)
        if why:
            _refuse("network", event, _describe_addr(args), why)
        return
    if event == "ctypes.call_function":
        if not _active("input"):
            return
        ptr = args[0] if args else None
        name = _input_fn_ptrs.get(ptr) or _message_fn_ptrs.get(ptr)
        if not name:
            return
        call_args = args[1] if len(args) > 1 and isinstance(
            args[1], tuple) else ()
        why = input_verdict(name, call_args, foreign=_foreign_window)
        if why:
            _refuse("input", f"user32.{name}", name, why)
        return
    if not _active("probe"):
        return
    if event in ("os.startfile", "os.startfile/2"):
        # (path, operation[, arguments, cwd, show_cmd]) - the Windows shell
        # launch: raised BEFORE ShellExecute, so a refusal opens nothing.
        target = _as_text(args[0]) if args else ""
        op = _as_text(args[1]) if len(args) > 1 and args[1] else "open"
        _refuse("probe", event, target[:160] or "<path>",
                f"os.startfile ({op}) is a desktop file launcher (it opens "
                f"the file in its app on the owner's desktop)")
        return
    if event == "subprocess.Popen":
        executable, cmd = (args + (None, None))[:2]
    elif event == "os.system":
        executable, cmd = None, args[0] if args else ""
    elif event == "os.spawn":             # (mode, path, args, env)
        executable, cmd = (args[1] if len(args) > 1 else None,
                           args[2] if len(args) > 2 else None)
    else:                                 # os.posix_spawn (path, argv, env)
        executable, cmd = (args[0] if args else None,
                           args[1] if len(args) > 1 else None)
    why = probe_verdict(executable, cmd)
    if why:
        target = " ".join(_command_tokens(cmd))[:160] or _as_text(executable)
        _refuse("probe", event, target, why)


# ─── pywin32 (not audited: wrapped like the browser guard) ─────────────────

def _marked(obj) -> bool:
    depth = 0
    while obj is not None and depth < 12:
        depth += 1
        try:
            if object.__getattribute__(obj, _GUARD_MARK) is True:
                return True
        except Exception:  # noqa: BLE001
            pass
        try:
            nxt = object.__getattribute__(obj, "__wrapped__")
        except Exception:  # noqa: BLE001
            return False
        if nxt is obj:
            return False
        obj = nxt
    return False


def _wrap_pywin32(module, modname: str, name: str) -> None:
    try:
        real = getattr(module, name)
    except Exception:  # noqa: BLE001
        return
    if _marked(real):
        return

    def stub(*args, **kwargs):
        if _active("input"):
            try:
                why = input_verdict(name, args, foreign=_foreign_window)
            except Exception:  # noqa: BLE001 - a guard bug must not break it
                why = None
            if why:
                _refuse("input", f"{modname}.{name}", name, why)
        return real(*args, **kwargs)

    stub.__name__ = getattr(real, "__name__", name)
    stub.__doc__ = getattr(real, "__doc__", None)
    stub.__wrapped__ = real
    setattr(stub, _GUARD_MARK, True)
    try:
        setattr(module, name, stub)
    except Exception:  # noqa: BLE001
        pass


def _arm_pywin32() -> None:
    if sys.platform != "win32":
        return
    import importlib
    import importlib.util
    for modname, names in _PYWIN32_TARGETS:
        try:
            module = sys.modules.get(modname)
            if module is None:
                if importlib.util.find_spec(modname) is None:
                    continue
                module = importlib.import_module(modname)
        except Exception:  # noqa: BLE001 - pywin32 absent or broken: nothing to wrap
            continue
        for name in names:
            if hasattr(module, name):
                _wrap_pywin32(module, modname, name)


def unwrapped_pywin32() -> list[str]:
    """pywin32 input functions present in an imported module but NOT wrapped."""
    bad = []
    for modname, names in _PYWIN32_TARGETS:
        module = sys.modules.get(modname)
        if module is None or not hasattr(module, "__dict__"):
            continue
        for name in names:
            try:
                if hasattr(module, name) and not _marked(getattr(module, name)):
                    bad.append(f"{modname}.{name}")
            except Exception:  # noqa: BLE001
                bad.append(f"{modname}.{name}")
    return bad


# ─── un-audited real-world effects (COM / WinRT): wrapped on import ────────

def _wrap_effect(module, modname: str, cls_name, name: str) -> None:
    try:
        owner = getattr(module, cls_name) if cls_name else module
        real = getattr(owner, name)
    except Exception:  # noqa: BLE001 - an attribute this version lacks
        return
    if _marked(real):
        return
    label = f"{modname}.{cls_name + '.' if cls_name else ''}{name}"

    def stub(*args, **kwargs):
        if _active("input"):
            _refuse("input", label, label,
                    "a real effect on the owner's speakers / media session "
                    "(a COM / WinRT call no audit event covers)")
        return real(*args, **kwargs)

    stub.__name__ = getattr(real, "__name__", name)
    stub.__doc__ = getattr(real, "__doc__", None)
    stub.__wrapped__ = real
    setattr(stub, _GUARD_MARK, True)
    try:
        raw = owner.__dict__.get(name) if cls_name else None
        setattr(owner, name,
                staticmethod(stub) if isinstance(raw, staticmethod) else stub)
    except Exception:  # noqa: BLE001
        pass


def _arm_effects_in(modname: str, module) -> None:
    for mod, cls_name, name in _EFFECT_TARGETS:
        if mod == modname:
            _wrap_effect(module, modname, cls_name, name)
    _effect_modules[modname] = module


def _arm_effects() -> None:
    """Wrap the targets of every effect module that is already imported (the
    import hook covers the ones imported later) and install that hook once."""
    for modname in _EFFECT_MODULES:
        module = sys.modules.get(modname)
        if module is not None and getattr(module, "__spec__", None) is not None:
            _arm_effects_in(modname, module)
    if not any(_marked(f) for f in sys.meta_path):
        sys.meta_path.insert(0, _EffectImportHook())


class _EffectLoader:
    """Runs the real loader, then wraps the module's effect targets. Every
    other loader attribute (get_source, resource readers ...) is the real
    loader's."""

    def __init__(self, real, name: str):
        self._real = real
        self._name = name

    def create_module(self, spec):
        return self._real.create_module(spec)

    def exec_module(self, module):
        self._real.exec_module(module)
        try:
            _arm_effects_in(self._name, module)
        except Exception:  # noqa: BLE001 - a guard bug must not break an import
            pass

    def __getattr__(self, attr):
        return getattr(self._real, attr)


class _EffectImportHook:
    """A ``sys.meta_path`` entry that finds NOTHING itself: for an effect
    module it asks the finders after it, then swaps in an _EffectLoader so the
    real module is wrapped the moment it has executed. A module a test put in
    ``sys.modules`` never reaches a finder, so a fake is never wrapped."""

    def find_spec(self, name, path=None, target=None):
        if name not in _EFFECT_MODULES or getattr(_finding, "on", False):
            return None
        _finding.on = True
        try:
            spec = None
            for finder in list(sys.meta_path):
                if finder is self or _marked(finder):
                    continue
                find = getattr(finder, "find_spec", None)
                if find is None:
                    continue
                spec = find(name, path, target)
                if spec is not None:
                    break
        except Exception:  # noqa: BLE001 - let the normal import machinery try
            return None
        finally:
            _finding.on = False
        if spec is None or not hasattr(getattr(spec, "loader", None),
                                       "exec_module"):
            return None
        spec.loader = _EffectLoader(spec.loader, name)
        return spec


setattr(_EffectImportHook, _GUARD_MARK, True)


def unwrapped_effects() -> list[str]:
    """Effect targets of a REAL module this guard wrapped that are NOT
    wrapped right now (a fake module in sys.modules is not checked)."""
    bad = []
    for modname, cls_name, name in _EFFECT_TARGETS:
        module = _effect_modules.get(modname)
        if module is None or sys.modules.get(modname) is not module:
            continue
        try:
            owner = getattr(module, cls_name) if cls_name else module
            if not _marked(getattr(owner, name)):
                bad.append(f"{modname}.{cls_name + '.' if cls_name else ''}{name}")
        except Exception:  # noqa: BLE001
            continue
    return bad


def _resolve_user32() -> None:
    """Function-pointer -> name tables for the audited ctypes calls."""
    if sys.platform != "win32" or _input_fn_ptrs:
        return
    try:
        import ctypes
        user32 = ctypes.WinDLL("user32")
    except Exception:  # noqa: BLE001 - no user32 (or no ctypes): nothing to watch
        return
    for table, names in ((_input_fn_ptrs, _INPUT_FUNCTIONS + _WINDOW_FUNCTIONS),
                         (_message_fn_ptrs, _MESSAGE_FUNCTIONS)):
        for name in names:
            try:
                ptr = ctypes.cast(getattr(user32, name), ctypes.c_void_p).value
            except Exception:  # noqa: BLE001 - an export this Windows lacks
                continue
            if ptr:
                table[ptr] = name


# ─── opt-in ────────────────────────────────────────────────────────────────

@contextlib.contextmanager
def allow(*guards: str, reason: str = ""):
    """Let the named guards (default: all three) through while the block is
    open - on every thread. Also a decorator. ``reason`` is documentation."""
    names = guards or GUARDS
    for g in names:
        if g not in _opt_in:
            raise ValueError(f"unknown guard {g!r} (one of {GUARDS})")
    with _lock:
        for g in names:
            _opt_in[g] += 1
    try:
        yield
    finally:
        with _lock:
            for g in names:
                _opt_in[g] -= 1


# ─── install / status / report ─────────────────────────────────────────────

_PROBE_EVENTS = {
    "network": ("socket.connect", (None, ("192.0.2.1", 9))),
    "probe": ("subprocess.Popen", ("nvidia-smi", ["nvidia-smi"], None, None)),
}


def unarmed_guards() -> list[str]:
    """Which guards do NOT currently refuse, answered BY BEHAVIOUR: a
    synthetic audit event per guard (no I/O - ``sys.audit`` only runs the
    hooks) must come back refused. An escape-hatched or opted-in guard is
    reported, because it is not refusing."""
    bad: list[str] = []
    probes = dict(_PROBE_EVENTS)
    cursor = [p for p, n in _input_fn_ptrs.items() if n == "SetCursorPos"]
    if cursor:        # (a box with no user32 has no input calls to watch)
        probes["input"] = ("ctypes.call_function", (cursor[0], (0, 0)))
    _probing.on = True
    try:
        for guard in GUARDS:
            if guard not in probes:
                continue              # input on a box with no user32 to watch
            event, args = probes[guard]
            try:
                sys.audit(event, *args)
            except (NetworkGuardError, InputGuardError, ProbeGuardError):
                continue
            except Exception:  # noqa: BLE001 - someone else's hook refused it
                pass
            bad.append(guard)
    finally:
        _probing.on = False
    if "input" not in bad and (unwrapped_pywin32() or unwrapped_effects()):
        bad.append("input")
    return bad


def is_armed() -> bool:
    return _installed[0] and not unarmed_guards()


def banner() -> str:
    return _banner[0]


def install(record_only: bool = False, *, env: dict | None = None,
            quiet: bool = False) -> bool:
    """Arm the three guards. Returns True when every guard refuses.

    Idempotent and REPAIRING: a later call re-wraps any pywin32 function a
    test displaced. Never raises. Prints one banner line on the first call."""
    try:
        if _installed[0]:
            _arm_pywin32()
            _arm_effects()
            return not _disabled and not _record_only[0]
        _record_only[0] = bool(record_only)
        _disabled.clear()
        for g in GUARDS:
            if _env_allows(g, env):
                _disabled.add(g)
        try:
            _own_names.add(_host_text(socket.gethostname()))
        except Exception:  # noqa: BLE001
            pass
        _resolve_user32()
        _arm_pywin32()
        _arm_effects()
        if not _hook_added[0]:
            sys.addaudithook(_hook)
            _hook_added[0] = True
        _installed[0] = True
        if not _atexit_registered[0]:
            atexit.register(_print_summary_at_exit)
            _atexit_registered[0] = True
        on = [g for g in GUARDS if g not in _disabled]
        off = [g for g in GUARDS if g in _disabled]
        if _record_only[0]:
            text = ("[hermetic-guard] RECORD-ONLY - network / input / probe "
                    "reaches are ALLOWED THROUGH and merely logged")
        elif not on:
            text = "[hermetic-guard] NOT armed"
        else:
            text = (f"[hermetic-guard] armed - {', '.join(on) or 'nothing'} "
                    f"refused and recorded (live ports "
                    f"{', '.join(str(p) for p in sorted(LIVE_SERVICE_PORTS))})")
        if off:
            text += ("; DISABLED via " + ", ".join(ENV_ESCAPES[g] for g in off)
                     + " - this run CAN reach the real "
                     + " / ".join(off))
        _banner[0] = text
        if not quiet:
            print(text, flush=True)
        return not off and not _record_only[0]
    except Exception as exc:  # noqa: BLE001 - a guard must never fail a run
        _banner[0] = (f"[hermetic-guard] WARNING: could NOT arm "
                      f"({type(exc).__name__}: {exc}) - this run CAN reach "
                      f"the network, real input and live hardware")
        if not quiet:
            print(_banner[0], flush=True)
        return False


def summary() -> str:
    """The atexit report, one block per guard that refused something."""
    lines: list[str] = []
    for guard in GUARDS:
        rows = refusals(guard)
        if not rows:
            continue
        tag = _TAGS[guard]
        who = {r.test_id or r.call_site or "<unknown>" for r in rows}
        sites: dict[tuple, int] = {}
        for r in rows:
            key = (r.test_id, r.api, r.target, r.call_site)
            sites[key] = sites.get(key, 0) + 1
        verb = "RECORDED" if _record_only[0] else "refused"
        lines.append(f"{tag} {verb} {len(rows)} real {guard} reach(es) "
                     f"from {len(who)} test(s)")
        shown = sorted(sites.items(), key=lambda kv: (-kv[1], kv[0]))
        for (test_id, api, target, site), count in shown[:40]:
            lines.append(f"{tag}   x{count} {test_id or '<unknown test>'} -> "
                         f"{api} {target} at {site or '<unknown site>'}")
        if len(shown) > 40:
            lines.append(f"{tag}   ... and {len(shown) - 40} more")
        lines.append(f"{tag}   (fake the boundary in the test, or opt in with "
                     f"tools.hermetic_guard.allow({guard!r}))")
    return "\n".join(lines)


def _print_summary_at_exit() -> None:
    try:
        text = summary()
        if text:
            print(text, flush=True)
    except Exception:  # noqa: BLE001 - an atexit hook must never explode
        pass


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    sys.exit(0 if install() and is_armed() else 1)
