"""
skills/code_executor.py — Open-Interpreter-style Python sandbox.

Gives Claude a `run_python(code)` action that executes arbitrary Python in
an isolated subprocess and returns the captured stdout/stderr. Lets JARVIS
answer one-off computational questions ('graph CPU temp over the last hour
from the log', 'convert this CSV to JSON', 'what's 7! divided by 137')
without writing a dedicated skill for every micro-task.

Two execution backends, chosen at call time:
  • subprocess (default) — fresh `python -c` per call. No state across
    calls. Bulletproof: a runaway script can't poison the parent. 30 s
    timeout enforced via process kill.
  • jupyter      — reuse a single IPython kernel via `jupyter_client` so
    variables persist call-to-call ('a = load_csv(...)' then later
    'print(a.describe())'). Activated by setting CODE_EXECUTOR_BACKEND
    to 'jupyter' or by passing backend='jupyter' through the public
    helpers. Falls back to subprocess if jupyter_client isn't installed.

The pre-installed scientific stack (pandas, numpy, matplotlib, requests,
pillow) is whatever the parent Python environment has. The subprocess
inherits PYTHONPATH from the parent so any pip-installed packages are
visible. Credential-shaped env vars (anything matching API_KEY, _TOKEN,
SECRET, PASSWORD, …) are stripped from the inherited environment so an
injected `run_python` snippet can't read them — see _sandbox_env. Note
this is mitigation, not isolation: the snippet still has unrestricted
filesystem and network access. True sandboxing (Docker, seccomp) is a
longer-term hardening step.

Registered actions
------------------
    run_python <code>          execute Python, return stdout/stderr
    python      <code>         alias
    eval_python <code>         alias
    compute     <code>         alias matching 'JARVIS, compute X' phrasing
    reset_kernel               wipe persistent jupyter state (if engaged)

Config
------
    CODE_EXECUTOR_BACKEND      'subprocess' (default) | 'jupyter'
    CODE_EXECUTOR_TIMEOUT      seconds, default 30
"""
from __future__ import annotations

import atexit
import os
import re
import subprocess
import sys
import tempfile
import threading
import time

_DEFAULT_TIMEOUT = 30
_MAX_OUTPUT_CHARS = 8000  # truncate noisy output before returning to the LLM
_CODE_FENCE_RE = re.compile(r"^\s*```(?:python|py)?\s*\n?(.*?)\n?```\s*$",
                            re.DOTALL | re.IGNORECASE)

# Substring-based denylist applied (case-insensitively) to env-var names
# before handing the environment to a sandbox subprocess or jupyter kernel.
# A prompt-injection that smuggles `run_python` could otherwise exfiltrate
# any credential parent JARVIS holds (ANTHROPIC_API_KEY, GOVEE_API_KEY,
# PORCUPINE_ACCESS_KEY, TELEGRAM_BOT_TOKEN, GITHUB_PERSONAL_ACCESS_TOKEN, …).
# Substrings catch future additions without requiring an explicit allowlist
# update each time a new credential is wired in.
_SENSITIVE_ENV_PATTERNS = (
    "API_KEY", "ACCESS_KEY", "ACCESS_TOKEN", "BOT_TOKEN", "AUTH_TOKEN",
    "_TOKEN", "SECRET", "PASSWORD", "PASSWD", "PRIVATE_KEY", "CREDENTIAL",
)


def _sandbox_env() -> dict[str, str]:
    """Return a copy of os.environ with credential-shaped vars removed."""
    out: dict[str, str] = {}
    for k, v in os.environ.items():
        ku = k.upper()
        if any(p in ku for p in _SENSITIVE_ENV_PATTERNS):
            continue
        out[k] = v
    return out

# Lazy-loaded jupyter kernel (per-process singleton).
_kernel_lock = threading.Lock()
_kernel = None        # KernelManager
_kernel_client = None # BlockingKernelClient
_atexit_registered = False  # one-shot: register _shutdown_kernel on first start


def _bobert():
    """Resolve the bobert_companion module for config lookup without a
    circular import (skills are loaded from inside it)."""
    return sys.modules.get("__main__") or sys.modules.get("bobert_companion")


def _resolve(env_name: str, attr_name: str, default):
    val = os.environ.get(env_name)
    if val is not None and val != "":
        return val
    b = _bobert()
    if b is not None and hasattr(b, attr_name):
        v = getattr(b, attr_name)
        if v is not None and v != "":
            return v
    return default


def _config_timeout() -> int:
    try:
        return max(1, int(_resolve("CODE_EXECUTOR_TIMEOUT",
                                   "CODE_EXECUTOR_TIMEOUT", _DEFAULT_TIMEOUT)))
    except (TypeError, ValueError):
        return _DEFAULT_TIMEOUT


def _config_backend() -> str:
    b = str(_resolve("CODE_EXECUTOR_BACKEND",
                     "CODE_EXECUTOR_BACKEND", "subprocess")).strip().lower()
    return b if b in ("subprocess", "jupyter") else "subprocess"


def _strip_fence(code: str) -> str:
    """Strip ```python fences the LLM sometimes wraps code in."""
    m = _CODE_FENCE_RE.match(code)
    return m.group(1) if m else code


def _truncate(text: str) -> str:
    if len(text) <= _MAX_OUTPUT_CHARS:
        return text
    head = text[: _MAX_OUTPUT_CHARS // 2]
    tail = text[-_MAX_OUTPUT_CHARS // 2 :]
    dropped = len(text) - len(head) - len(tail)
    return f"{head}\n... [{dropped} chars truncated] ...\n{tail}"


def _format_result(stdout: str, stderr: str, rc: int | None, dt: float) -> str:
    """Build the single string returned to the LLM. Keeps stdout primary,
    appends stderr only when present, and always prefixes a one-line
    status header so the LLM can tell success from failure at a glance."""
    parts: list[str] = []
    if rc is None:
        parts.append(f"[timeout after {dt:.1f}s — process killed]")
    elif rc != 0:
        parts.append(f"[exited with code {rc} in {dt:.1f}s]")
    else:
        parts.append(f"[ok in {dt:.1f}s]")

    out = (stdout or "").rstrip()
    err = (stderr or "").rstrip()
    if out:
        parts.append("stdout:\n" + _truncate(out))
    if err:
        parts.append("stderr:\n" + _truncate(err))
    if not out and not err and rc == 0:
        parts.append("(no output)")
    return "\n".join(parts)


# ── subprocess backend ─────────────────────────────────────────────────
def _kill_process_tree(pid: int) -> None:
    """Kill `pid` AND all of its descendants. A snippet that spawns a
    DETACHED grandchild (e.g. via subprocess.Popen / os.spawn) outlives a
    plain proc.kill(), which only reaps the direct child — so on timeout we
    tear down the whole tree. Prefers psutil; falls back to Windows
    `taskkill /F /T`. Best-effort + exception-safe."""
    # Preferred: psutil walks the descendant tree portably.
    try:
        import psutil  # type: ignore[import-not-found]
        try:
            parent = psutil.Process(pid)
        except psutil.NoSuchProcess:
            return
        children = parent.children(recursive=True)
        for child in children:
            try:
                child.kill()
            except Exception:
                pass
        try:
            parent.kill()
        except Exception:
            pass
        # Reap so we don't leave zombies and so VRAM/handles release promptly.
        try:
            psutil.wait_procs(children + [parent], timeout=3)
        except Exception:
            pass
        return
    except Exception:
        pass

    # Fallback: Windows taskkill with /T (tree) /F (force).
    if sys.platform == "win32":
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                capture_output=True,
                creationflags=subprocess.CREATE_NO_WINDOW,
                timeout=10,
            )
        except Exception:
            pass
    else:
        # POSIX last resort: SIGKILL the single pid (no psutil → no tree).
        try:
            os.kill(pid, 9)
        except Exception:
            pass


def _run_subprocess(code: str, timeout: int) -> str:
    """Write the snippet to a temp file and run it under the current Python.

    Using a temp file (vs `python -c`) makes tracebacks readable — they
    cite real line numbers and the file path, which the LLM can use to
    fix its own bugs on retry."""
    # NamedTemporaryFile on Windows can't be reopened while still open in
    # the writer, so we explicitly close before spawning and clean up
    # ourselves in finally.
    fd, path = tempfile.mkstemp(suffix=".py", prefix="jarvis_exec_", text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(code)
        t0 = time.time()
        # Popen (not subprocess.run) so we hold the pid — on timeout we kill
        # the whole process tree, not just the direct child, so a detached
        # grandchild can't survive the deadline. The normal-completion path
        # below is functionally identical to the old subprocess.run().
        try:
            proc = subprocess.Popen(
                [sys.executable, path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                close_fds=True,
                env=_sandbox_env(),
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            )
        except Exception as e:
            return f"[execution failed: {type(e).__name__}: {e}]"

        try:
            out, err = proc.communicate(timeout=timeout)
            dt = time.time() - t0
            return _format_result(out, err, proc.returncode, dt)
        except subprocess.TimeoutExpired:
            # Tear down the entire tree, then drain whatever output we got.
            _kill_process_tree(proc.pid)
            dt = time.time() - t0
            try:
                out, err = proc.communicate(timeout=5)
            except Exception:
                out, err = "", ""
            return _format_result(out or "", err or "", None, dt)
        except Exception as e:
            try:
                _kill_process_tree(proc.pid)
            except Exception:
                pass
            return f"[execution failed: {type(e).__name__}: {e}]"
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


# ── jupyter backend ────────────────────────────────────────────────────
def _ensure_kernel():
    """Spin up (or reuse) the persistent IPython kernel. Returns the
    BlockingKernelClient, or None if jupyter_client isn't installed."""
    global _kernel, _kernel_client, _atexit_registered
    with _kernel_lock:
        if _kernel_client is not None:
            return _kernel_client
        try:
            from jupyter_client.manager import KernelManager
        except ImportError:
            return None
        try:
            km = KernelManager(kernel_name="python3")
            # Strip credential-shaped env vars before the kernel inherits
            # them — same defense as the subprocess backend.
            km.start_kernel(env=_sandbox_env())
            kc = km.blocking_client()
            kc.start_channels()
            kc.wait_for_ready(timeout=10)
            _kernel = km
            _kernel_client = kc
            # The kernel is a subprocess with zmq channel threads; reset_kernel
            # is the only manual teardown, so a normal JARVIS exit would orphan
            # them. Register the shutdown once (first start) so interpreter exit
            # reaps the kernel cleanly.
            if not _atexit_registered:
                atexit.register(_shutdown_kernel)
                _atexit_registered = True
            return kc
        except Exception as e:
            print(f"  [code-exec] jupyter kernel failed to start: {e}")
            _kernel = None
            _kernel_client = None
            return None


def _run_jupyter(code: str, timeout: int) -> str:
    kc = _ensure_kernel()
    if kc is None:
        # Fall back transparently — better to run statelessly than to
        # refuse the call when the user just wanted an answer.
        return _run_subprocess(code, timeout)

    t0 = time.time()
    msg_id = kc.execute(code)
    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    status = "ok"
    deadline = t0 + timeout

    global _kernel, _kernel_client
    while True:
        remaining = deadline - time.time()
        if remaining <= 0:
            # Interrupt the kernel so the next call starts cleanly — but
            # only if the kernel process is still alive. If it died between
            # starting and timing out, the OS may have recycled its PID for
            # an unrelated process; signalling it would hit the wrong one.
            kernel_alive = True
            try:
                kernel_alive = bool(_kernel and _kernel.is_alive())
            except Exception:
                kernel_alive = False
            if kernel_alive:
                try:
                    _kernel.interrupt_kernel()
                except Exception:
                    pass
                return _format_result(
                    "".join(stdout_parts), "".join(stderr_parts), None,
                    time.time() - t0,
                )
            # Kernel is gone — drop the stale handles and fall back to a
            # one-shot subprocess so the caller still gets an answer.
            with _kernel_lock:
                _kernel = None
                _kernel_client = None
            return _run_subprocess(code, timeout)
        try:
            msg = kc.get_iopub_msg(timeout=min(1.0, remaining))
        except Exception:
            continue
        if msg.get("parent_header", {}).get("msg_id") != msg_id:
            continue
        mtype = msg["msg_type"]
        content = msg.get("content", {})
        if mtype == "stream":
            (stdout_parts if content.get("name") == "stdout"
             else stderr_parts).append(content.get("text", ""))
        elif mtype in ("execute_result", "display_data"):
            data = content.get("data", {})
            text = data.get("text/plain")
            if text:
                stdout_parts.append(text + "\n")
        elif mtype == "error":
            status = "error"
            stderr_parts.append("\n".join(content.get("traceback", [])) + "\n")
        elif mtype == "status" and content.get("execution_state") == "idle":
            break

    rc = 0 if status == "ok" else 1
    return _format_result("".join(stdout_parts), "".join(stderr_parts),
                          rc, time.time() - t0)


def _shutdown_kernel() -> str:
    global _kernel, _kernel_client
    with _kernel_lock:
        if _kernel_client is None:
            return "no kernel was running, sir"
        try:
            _kernel_client.stop_channels()
        except Exception:
            pass
        try:
            _kernel.shutdown_kernel(now=True)
        except Exception:
            pass
        _kernel = None
        _kernel_client = None
    return "kernel reset, sir — variables cleared"


# ── public actions ─────────────────────────────────────────────────────
def run_python(code: str = "") -> str:
    code = _strip_fence((code or "").strip())
    if not code:
        return ("format: run_python, <python code>  "
                "(multi-line is fine; result is stdout/stderr from a "
                "30 s sandbox)")
    timeout = _config_timeout()
    backend = _config_backend()
    if backend == "jupyter":
        return _run_jupyter(code, timeout)
    return _run_subprocess(code, timeout)


def reset_kernel(_: str = "") -> str:
    return _shutdown_kernel()


def register(actions: dict):
    actions["run_python"]   = run_python
    actions["python"]       = run_python
    actions["eval_python"]  = run_python
    actions["compute"]      = run_python
    actions["reset_kernel"] = reset_kernel
