"""Starts, watches and stops the hand tracker service (tools/hand_tracker_service.py).

The service runs in its OWN venv (MediaPipe + websockets + OpenCV 5, none of
which belong in JARVIS's environment), so JARVIS only ever sees a child
process, a heartbeat file and a websocket on 127.0.0.1.

  * start()  spawns ``HAND_TRACK_PYTHON tools/hand_tracker_service.py ...`` as
             a hidden, windowless child (same recipe as the clone voice
             server) and starts the keeper thread. Prerequisites are checked
             first and an honest reason is returned when one is missing (venv
             python, the service script, the .task model). Idempotent.
  * keeper   once a second: touches data/.handtrack_jarvis_alive (the service
             exits when this stops, and when JARVIS's pid is gone), and
             restarts a service that died - at most MAX_RESTARTS times in
             RESTART_WINDOW_S, then it gives up with one log line.
  * stop()   asks the child to exit by removing its lifeline (it also gets a
             terminate after a short grace) and removes the heartbeat files.
             Hand tracking off means NOTHING runs and no frames are copied.
  * Kinect pump (2026-11-17): start() / stop() also start / stop
             core.handtrack_depth's Kinect pump, the thread that hands the
             tracker Kinect colour frames at the sensor's rate (it stands still
             unless the Kinect is the selected camera and the tracker is alive).
  * --camera / --det-conf / --presence-conf / --track-conf (2026-11-17): the
             camera kind the tracker starts with (its size / score limits;
             each frame's header can still say otherwise) and MediaPipe's three
             confidences from HAND_TRACK_DET_CONF / _PRESENCE_CONF /
             _TRACKING_CONF.

Never raises into callers. Stdlib only. Tests: tests/test_hand_tracker_launcher.py
(fake Popen, fake clock; no real subprocess).
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from typing import Callable, Optional

from core import handtrack_frames as _frames

MAX_RESTARTS = 3
RESTART_WINDOW_S = 600.0
KEEPER_PERIOD_S = 1.0
STOP_GRACE_S = 3.0
_STRIP_ENV = ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONEXECUTABLE",
              "__PYVENV_LAUNCHER__")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVICE_SCRIPT = os.path.join(_ROOT, "tools", "hand_tracker_service.py")
LOG_PATH = os.path.join(_ROOT, "data", "logs", "hand_tracker.log")


def _cfg(name: str, default):
    try:
        from core import config
        return getattr(config, name, default)
    except Exception:
        return default


CAMERA_KINDS = ("webcam", "kinect")
CONF_DEFAULT = 0.5
CONF_MIN, CONF_MAX = 0.05, 0.95


def build_argv(python: str, script: str, model: str, port: int,
               jarvis_pid: int, alive: str, jarvis_alive: str,
               camera: str = "", det_conf=None, presence_conf=None,
               track_conf=None):
    """The service's command line (a list: no shell). ``camera`` (one of
    CAMERA_KINDS) and the three confidences are only passed when given."""
    argv = [python, "-u", script, "--host", "127.0.0.1", "--port", str(int(port)),
            "--model", model, "--alive", alive, "--jarvis-pid", str(int(jarvis_pid)),
            "--jarvis-alive", jarvis_alive]
    if camera in CAMERA_KINDS:
        argv += ["--camera", camera]
    for flag, val in (("--det-conf", det_conf), ("--presence-conf", presence_conf),
                      ("--track-conf", track_conf)):
        if val is not None:
            argv += [flag, f"{float(val):.2f}"]
    return argv


def camera_kind() -> str:
    """"kinect" when the tracker's frames will come from the Kinect
    (HAND_TRACK_CAMERA, or a Kinect hologram that is up), else "webcam".
    Live 11-17: the desktop runs HAND_TRACK_CAMERA = "kinect", whose wide view
    makes hands ~0.065 of the frame height at 1.2 m - under the webcam-tuned
    0.075 floor, so palm_hold / pinch_hold almost never fired. Never raises."""
    try:
        from core import handtrack_depth
        setting = str(_cfg("HAND_TRACK_CAMERA", "primary") or "primary")
        eff = handtrack_depth.effective_camera_setting(setting)
        return "kinect" if handtrack_depth.is_kinect_setting(eff) else "webcam"
    except Exception:
        return "webcam"


def conf_value(name: str) -> float:
    """A MediaPipe confidence setting, clamped to CONF_MIN..CONF_MAX; a bad
    value falls back to CONF_DEFAULT (0.5, MediaPipe's own default)."""
    try:
        v = float(_cfg(name, CONF_DEFAULT))
    except Exception:
        return CONF_DEFAULT
    if v != v:                                       # NaN
        return CONF_DEFAULT
    return min(CONF_MAX, max(CONF_MIN, v))


class _DepthPumpHook:
    """core.handtrack_depth's Kinect pump, imported when first used (the
    module-level LAUNCHER's default; tests pass their own or none)."""

    @staticmethod
    def start() -> bool:
        try:
            from core import handtrack_depth
            return bool(handtrack_depth.start_pump())
        except Exception:
            return False

    @staticmethod
    def stop() -> bool:
        try:
            from core import handtrack_depth
            return bool(handtrack_depth.stop_pump())
        except Exception:
            return False


class HandTrackerLauncher:
    def __init__(self, *, popen: Optional[Callable] = None,
                 clock: Callable[[], float] = time.monotonic,
                 pid: Optional[int] = None, log: Callable[[str], None] = print,
                 exists: Callable[[str], bool] = os.path.isfile,
                 alive_file: Optional[str] = None,
                 jarvis_file: Optional[str] = None,
                 log_path: Optional[str] = None,
                 start_keeper: bool = True,
                 kinect_pump=None):
        self._popen = popen
        # start() / stop() objects for the Kinect pump (None = no pump: tests).
        self._kinect_pump = kinect_pump
        self._clock = clock
        self._pid = pid if pid is not None else os.getpid()
        self._log = log
        self._exists = exists
        self._alive_file = alive_file or _frames.alive_path()
        self._jarvis_file = jarvis_file or _frames.jarvis_alive_path()
        self._start_keeper = start_keeper
        self._log_path = log_path or LOG_PATH
        self._mu = threading.RLock()
        self._proc = None
        self._wanted = False
        self._restarts: list = []
        self._gave_up = False
        self._last_error = ""
        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._argv = None
        self._started_at = 0.0

    # ── prerequisites ───────────────────────────────────────────────────
    def prerequisites(self) -> str:
        """'' when the service can be started, else why not (one sentence)."""
        py = str(_cfg("HAND_TRACK_PYTHON", ""))
        model = str(_cfg("HAND_TRACK_MODEL", ""))
        if not py or not self._exists(py):
            return ("the hand tracker's Python environment isn't installed "
                    "(HAND_TRACK_PYTHON)")
        if not self._exists(SERVICE_SCRIPT):
            return "the hand tracker script is missing"
        if not model or not self._exists(model):
            return "the hand model file isn't downloaded (HAND_TRACK_MODEL)"
        return ""

    # ── lifecycle ───────────────────────────────────────────────────────
    def start(self) -> tuple:
        """(ok, message). Starts the service unless it is already running."""
        with self._mu:
            self._wanted = True
            self._gave_up = False
            if self.running():
                self._pump_start()
                return True, "already running"
            why = self.prerequisites()
            if why:
                self._wanted = False
                self._last_error = why
                return False, why
            self._touch_lifeline()
            ok, msg = self._spawn()
            if not ok:
                self._wanted = False
                return False, msg
            self._ensure_keeper()
            self._pump_start()
            return True, "started"

    def stop(self) -> None:
        with self._mu:
            self._wanted = False
            proc = self._proc
            self._proc = None
        self._pump_stop()
        # Removing the lifeline makes the service exit by itself (it also
        # watches our pid); terminate only if it lingers.
        self._remove(self._jarvis_file)
        if proc is not None:
            try:
                deadline = self._clock() + STOP_GRACE_S
                while proc.poll() is None and self._clock() < deadline:
                    time.sleep(0.05)
                if proc.poll() is None:
                    proc.terminate()
            except Exception:
                pass
        self._remove(self._alive_file)
        self._log("  [hand-track] stopped")

    def running(self) -> bool:
        with self._mu:
            p = self._proc
        try:
            return p is not None and p.poll() is None
        except Exception:
            return False

    def wanted(self) -> bool:
        return self._wanted

    def gave_up(self) -> bool:
        return self._gave_up

    def last_error(self) -> str:
        return self._last_error

    # ── internals ───────────────────────────────────────────────────────
    def _pump_start(self) -> None:
        if self._kinect_pump is not None:
            try:
                self._kinect_pump.start()
            except Exception:
                pass

    def _pump_stop(self) -> None:
        if self._kinect_pump is not None:
            try:
                self._kinect_pump.stop()
            except Exception:
                pass

    def _touch_lifeline(self) -> None:
        try:
            os.makedirs(os.path.dirname(self._jarvis_file) or ".", exist_ok=True)
            with open(self._jarvis_file, "w", encoding="utf-8") as f:
                f.write(str(int(time.time())))
        except OSError:
            pass

    @staticmethod
    def _remove(path: str) -> None:
        try:
            os.remove(path)
        except OSError:
            pass

    def _spawn(self) -> tuple:
        argv = build_argv(str(_cfg("HAND_TRACK_PYTHON", "")), SERVICE_SCRIPT,
                          str(_cfg("HAND_TRACK_MODEL", "")),
                          int(_cfg("HAND_TRACK_PORT", 8793)), self._pid,
                          self._alive_file, self._jarvis_file,
                          camera=camera_kind(),
                          det_conf=conf_value("HAND_TRACK_DET_CONF"),
                          presence_conf=conf_value("HAND_TRACK_PRESENCE_CONF"),
                          track_conf=conf_value("HAND_TRACK_TRACKING_CONF"))
        self._argv = argv
        env = {k: v for k, v in os.environ.items() if k not in _STRIP_ENV}
        kw = {"stdin": subprocess.DEVNULL, "stderr": subprocess.STDOUT,
              "close_fds": True, "env": env, "cwd": _ROOT}
        if os.name == "nt":
            # Hidden console (not DETACHED_PROCESS: that makes the child
            # allocate a VISIBLE one) - zero windows.
            kw["creationflags"] = (getattr(subprocess, "CREATE_NO_WINDOW", 0)
                                   | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
        logf = None
        try:
            try:
                os.makedirs(os.path.dirname(self._log_path), exist_ok=True)
                logf = open(self._log_path, "wb")
                kw["stdout"] = logf
            except OSError:
                kw["stdout"] = subprocess.DEVNULL
            popen = self._popen or subprocess.Popen
            self._proc = popen(argv, **kw)
            self._started_at = self._clock()
            self._log("  [hand-track] tracker started (pid "
                      f"{getattr(self._proc, 'pid', '?')}); log: data/logs/hand_tracker.log")
            return True, "started"
        except Exception as e:
            self._proc = None
            self._last_error = f"{type(e).__name__}: {e}"
            self._log(f"  [hand-track] could not start the tracker: {self._last_error}")
            return False, f"it would not start ({type(e).__name__})"
        finally:
            if logf is not None:
                try:
                    logf.close()
                except Exception:
                    pass

    def _ensure_keeper(self) -> None:
        if not self._start_keeper:
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._keep, name="hand-track-keeper",
                                        daemon=True)
        self._thread.start()

    def _keep(self) -> None:
        while not self._stop_evt.wait(KEEPER_PERIOD_S):
            if not self.tick():
                return

    def tick(self) -> bool:
        """One keeper pass. False once tracking is no longer wanted."""
        with self._mu:
            if not self._wanted:
                return False
            self._touch_lifeline()
            if self.running():
                return True
            if self._gave_up:
                return True
            now = self._clock()
            self._restarts = [t for t in self._restarts if now - t < RESTART_WINDOW_S]
            code = None
            try:
                code = self._proc.poll() if self._proc is not None else None
            except Exception:
                pass
            if len(self._restarts) >= MAX_RESTARTS:
                self._gave_up = True
                self._last_error = "the tracker kept stopping"
                self._log(f"  [hand-track] the tracker stopped {MAX_RESTARTS} times in "
                          f"{int(RESTART_WINDOW_S // 60)} min - giving up until "
                          "hand tracking is switched on again")
                return True
            self._restarts.append(now)
            self._log(f"  [hand-track] the tracker exited (code {code}); restarting")
            self._spawn()
            return True

    # ── status ──────────────────────────────────────────────────────────
    def heartbeat(self) -> Optional[dict]:
        """The service's last status line (its heartbeat file) with ``age_s``,
        or None when it is not writing one."""
        age = _frames.file_age_s(self._alive_file)
        if age is None:
            return None
        try:
            with open(self._alive_file, "r", encoding="utf-8") as f:
                d = json.load(f)
            if not isinstance(d, dict):
                d = {}
        except Exception:
            d = {}
        d["age_s"] = age
        return d


LAUNCHER = HandTrackerLauncher(kinect_pump=_DepthPumpHook())
