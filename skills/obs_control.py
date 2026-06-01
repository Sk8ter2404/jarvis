"""
OBS Studio voice control skill for JARVIS.

Talks to a local OBS Studio instance over its built-in WebSocket API (v5,
default port 4455 since OBS 28). Uses the `obs-websocket-py` library
(PyPI: obs-websocket-py, import: obswebsocket).

Actions added:
  obs_start_recording        — begin a new recording
  obs_stop_recording         — finish the current recording
  obs_pause_recording        — pause/resume the current recording (toggle)
  obs_switch_scene <name>    — switch program output to the named scene
  obs_toggle_mute  <source>  — flip mute on the named audio input

Voice phrasing the LLM is expected to map onto these actions:
  "start recording"          → obs_start_recording
  "stop recording"           → obs_stop_recording
  "pause recording"          → obs_pause_recording
  "switch to gameplay scene" → obs_switch_scene  gameplay
  "mute the mic"             → obs_toggle_mute   Mic/Aux

Connection:
  Auto-connects to ws://127.0.0.1:4455 with the password from the
  OBS_PASSWORD env var (leave unset if OBS WebSocket auth is disabled).
  Each action opens a fresh short-lived connection so the skill survives
  OBS being closed and reopened without re-registering. If obs-websocket-py
  isn't installed, the actions still register but return an install hint
  instead of crashing on first call.

Config (env vars):
  OBS_PASSWORD   — WebSocket password (Tools → WebSocket Server Settings)
  OBS_HOST       — override default host (rare, defaults to 127.0.0.1)
  OBS_PORT       — override default port (rare, defaults to 4455)
"""
import os

_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 4455
# Per-call timeout (seconds) for the websocket handshake + request. OBS on
# the local loopback usually answers in <50 ms; 5 s is generous but still
# short enough that a misconfigured / closed OBS doesn't hang the voice loop.
_CALL_TIMEOUT = 5.0

try:
    from obswebsocket import obsws, requests as obs_requests
    _HAS_OBSWS = True
    _IMPORT_ERROR: Exception | None = None
except Exception as e:  # ImportError, or DLL-load issues on some boxes
    _HAS_OBSWS = False
    _IMPORT_ERROR = e
    obsws = None           # type: ignore[assignment]
    obs_requests = None    # type: ignore[assignment]


def _config() -> tuple[str, int, str]:
    """Read host/port/password from env every call so the user can change
    them without restarting JARVIS."""
    host = os.environ.get("OBS_HOST", _DEFAULT_HOST).strip() or _DEFAULT_HOST
    try:
        port = int(os.environ.get("OBS_PORT", "").strip() or _DEFAULT_PORT)
    except ValueError:
        port = _DEFAULT_PORT
    password = os.environ.get("OBS_PASSWORD", "")
    return host, port, password


def _missing_dep_msg() -> str:
    hint = f" ({_IMPORT_ERROR})" if _IMPORT_ERROR else ""
    return ("obs-websocket-py is not installed, sir — "
            f"pip install obs-websocket-py to enable OBS control{hint}")


def _with_ws(fn):
    """Open a short-lived WebSocket connection to OBS, run `fn(ws)`, and
    return its result. Translates connection / auth failures into a tidy
    one-line string suitable for TTS rather than letting a traceback leak
    to the user."""
    if not _HAS_OBSWS:
        return _missing_dep_msg()
    host, port, password = _config()
    ws = obsws(host, port, password, timeout=_CALL_TIMEOUT)
    try:
        ws.connect()
    except Exception as e:
        # Most common cause: OBS not running, or WebSocket Server disabled,
        # or wrong password. Surface a short reason rather than the raw
        # exception class — the user is going to hear this.
        msg = str(e) or e.__class__.__name__
        if "auth" in msg.lower() or "password" in msg.lower():
            return ("OBS rejected the password, sir — check OBS_PASSWORD "
                    "matches Tools → WebSocket Server Settings.")
        return (f"I couldn't reach OBS at {host}:{port}, sir — "
                "is OBS open with the WebSocket server enabled?")
    try:
        return fn(ws)
    finally:
        try:
            ws.disconnect()
        except Exception:
            pass


# ── action implementations ─────────────────────────────────────────────
def obs_start_recording(_: str = "") -> str:
    def _go(ws):
        try:
            ws.call(obs_requests.StartRecord())
        except Exception as e:
            # OBS replies with a typed error if recording is already running;
            # treat that as a friendly no-op rather than a failure.
            if "already" in str(e).lower():
                return "OBS is already recording, sir."
            return f"OBS refused to start recording: {e}"
        return "Recording, sir."
    return _with_ws(_go)


def obs_stop_recording(_: str = "") -> str:
    def _go(ws):
        try:
            ws.call(obs_requests.StopRecord())
        except Exception as e:
            if "not" in str(e).lower() and "record" in str(e).lower():
                return "OBS isn't currently recording, sir."
            return f"OBS refused to stop recording: {e}"
        return "Recording stopped, sir."
    return _with_ws(_go)


def obs_pause_recording(_: str = "") -> str:
    """Toggle pause/resume on the active recording. OBS distinguishes
    PauseRecord and ResumeRecord, so we check current state first and
    issue whichever transition applies. If no recording is running we
    say so rather than silently doing nothing."""
    def _go(ws):
        try:
            status = ws.call(obs_requests.GetRecordStatus())
        except Exception as e:
            return f"OBS didn't answer about recording state: {e}"
        # obs-websocket-py exposes response fields via .datain — the v5
        # GetRecordStatus reply carries outputActive + outputPaused.
        data = getattr(status, "datain", None) or {}
        active = bool(data.get("outputActive"))
        paused = bool(data.get("outputPaused"))
        if not active:
            return "OBS isn't recording right now, sir."
        try:
            if paused:
                ws.call(obs_requests.ResumeRecord())
                return "Recording resumed, sir."
            ws.call(obs_requests.PauseRecord())
            return "Recording paused, sir."
        except Exception as e:
            return f"OBS refused to toggle pause: {e}"
    return _with_ws(_go)


def obs_switch_scene(name: str = "") -> str:
    name = (name or "").strip().strip("'\"")
    if not name:
        return "format: obs_switch_scene, <scene name>"
    def _go(ws):
        # Confirm the scene exists so a typo gives a useful error rather
        # than OBS's terse 'ResourceNotFound'.
        try:
            scenes_resp = ws.call(obs_requests.GetSceneList())
        except Exception as e:
            return f"OBS didn't return its scene list: {e}"
        data = getattr(scenes_resp, "datain", None) or {}
        scene_names = [s.get("sceneName", "") for s in data.get("scenes", [])]
        # Case-insensitive resolution — voice transcripts won't preserve
        # the exact casing the user picked in OBS.
        match = next((s for s in scene_names if s.lower() == name.lower()), None)
        if match is None:
            # Try a substring match before giving up — handy when the user
            # says "switch to gameplay" and the scene is "Gameplay - Main".
            partial = [s for s in scene_names if name.lower() in s.lower()]
            if len(partial) == 1:
                match = partial[0]
            elif len(partial) > 1:
                return (f"Multiple scenes match '{name}', sir: "
                        f"{', '.join(partial)}. Be more specific.")
        if match is None:
            preview = ", ".join(scene_names[:6]) or "(none)"
            return (f"No scene called '{name}', sir. Available: {preview}"
                    + ("..." if len(scene_names) > 6 else ""))
        try:
            ws.call(obs_requests.SetCurrentProgramScene(sceneName=match))
        except Exception as e:
            return f"OBS refused to switch scene: {e}"
        return f"Scene switched to '{match}', sir."
    return _with_ws(_go)


def obs_toggle_mute(source: str = "") -> str:
    source = (source or "").strip().strip("'\"")
    if not source:
        return "format: obs_toggle_mute, <source name>"
    def _go(ws):
        # Resolve the source name case-insensitively against the input list
        # so 'mic' matches 'Mic/Aux'.
        try:
            inputs_resp = ws.call(obs_requests.GetInputList())
        except Exception as e:
            return f"OBS didn't return its input list: {e}"
        data = getattr(inputs_resp, "datain", None) or {}
        input_names = [i.get("inputName", "") for i in data.get("inputs", [])]
        match = next((n for n in input_names if n.lower() == source.lower()), None)
        if match is None:
            partial = [n for n in input_names if source.lower() in n.lower()]
            if len(partial) == 1:
                match = partial[0]
            elif len(partial) > 1:
                return (f"Multiple inputs match '{source}', sir: "
                        f"{', '.join(partial)}. Be more specific.")
        if match is None:
            preview = ", ".join(input_names[:6]) or "(none)"
            return (f"No audio source called '{source}', sir. "
                    f"Available: {preview}"
                    + ("..." if len(input_names) > 6 else ""))
        try:
            resp = ws.call(obs_requests.ToggleInputMute(inputName=match))
        except Exception as e:
            return f"OBS refused to toggle mute: {e}"
        # Report the new state if OBS told us; otherwise just confirm the
        # toggle landed.
        new_muted = (getattr(resp, "datain", None) or {}).get("inputMuted")
        if new_muted is True:
            return f"'{match}' is now muted, sir."
        if new_muted is False:
            return f"'{match}' is now live, sir."
        return f"Toggled mute on '{match}', sir."
    return _with_ws(_go)


def register(actions: dict):
    actions["obs_start_recording"] = obs_start_recording
    actions["obs_stop_recording"]  = obs_stop_recording
    actions["obs_pause_recording"] = obs_pause_recording
    actions["obs_switch_scene"]    = obs_switch_scene
    actions["obs_toggle_mute"]     = obs_toggle_mute
