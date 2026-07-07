"""Logic tests for skills/obs_control.py.

OBS control talks to a local OBS Studio over its websocket API. We never open
a real socket: the obs-websocket-py library is faked with a stub `obsws` whose
`.call()` returns canned response objects (with a `.datain` dict mirroring the
v5 protocol shape), so we can exercise the REAL logic the skill owns:
  * env-driven host/port/password config parsing,
  * the missing-dependency install hint,
  * connection-failure → tidy TTS string (incl. the auth-error branch),
  * scene resolution (exact, case-insensitive, unique-substring, ambiguous,
    unknown) and the SetCurrentProgramScene dispatch,
  * audio-source mute resolution + new-state reporting,
  * pause/resume toggling off the GetRecordStatus reply.

No real OBS, no sockets.
"""
from __future__ import annotations

import os
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated


class _Resp:
    """Stand-in for an obs-websocket-py response: `.datain` plus `.status`
    (obs-websocket-py never raises on a failed request — it returns normally
    with status=False and the reason in datain['comment'])."""
    def __init__(self, datain=None, status=True):
        self.datain = datain or {}
        self.status = status


class _FakeWS:
    """Records the request objects passed to .call() and returns scripted
    responses keyed by the request class name. Each request object created by
    obs_requests.<Name>(**kw) carries its class name + kwargs (see _fake_requests)."""
    def __init__(self, responses=None, raise_on=None):
        self._responses = responses or {}
        self._raise_on = raise_on or {}
        self.calls = []  # list of (name, kwargs)

    def connect(self):
        return None

    def disconnect(self):
        return None

    def call(self, req):
        name = req["__name__"]
        self.calls.append((name, {k: v for k, v in req.items() if k != "__name__"}))
        if name in self._raise_on:
            raise self._raise_on[name]
        resp = self._responses.get(name, _Resp())
        return resp


def _fake_requests():
    """A fake `obs_requests` module: each attribute is a constructor that
    returns a dict tagging its own name + kwargs, so _FakeWS can dispatch."""
    class _Req:
        def __getattr__(self, name):
            def _make(**kwargs):
                d = dict(kwargs)
                d["__name__"] = name
                return d
            return _make
    return _Req()


class ObsConfigAndDepTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("obs_control")

    def test_config_defaults(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            host, port, pw = self.mod._config()
        self.assertEqual(host, "127.0.0.1")
        self.assertEqual(port, 4455)
        self.assertEqual(pw, "")

    def test_config_reads_env(self):
        with mock.patch.dict(os.environ,
                             {"OBS_HOST": "10.0.0.5", "OBS_PORT": "4490",
                              "OBS_PASSWORD": "hunter2"}, clear=True):
            host, port, pw = self.mod._config()
        self.assertEqual(host, "10.0.0.5")
        self.assertEqual(port, 4490)
        self.assertEqual(pw, "hunter2")

    def test_config_bad_port_falls_back_to_default(self):
        with mock.patch.dict(os.environ, {"OBS_PORT": "not-a-number"}, clear=True):
            _host, port, _pw = self.mod._config()
        self.assertEqual(port, 4455)

    def test_missing_dependency_hint(self):
        # Force the "library not installed" branch regardless of the test box.
        with mock.patch.object(self.mod, "_HAS_OBSWS", False):
            out = self.actions["obs_start_recording"]("")
        self.assertIn("obs-websocket-py", out)
        self.assertIn("pip install", out)


class ObsConnectionTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("obs_control")

    def _install_ws(self, ws):
        """Patch the module so _with_ws builds our fake ws and the request
        factory yields tagging dicts."""
        self.mod._HAS_OBSWS = True
        return mock.patch.multiple(
            self.mod,
            obsws=mock.MagicMock(return_value=ws),
            obs_requests=_fake_requests(),
        )

    def test_connect_failure_is_friendly(self):
        ws = _FakeWS()
        ws.connect = mock.MagicMock(side_effect=OSError("connection refused"))
        with self._install_ws(ws):
            out = self.actions["obs_start_recording"]("")
        self.assertIn("couldn't reach OBS", out)
        self.assertIn("4455", out)

    def test_connect_auth_failure_message(self):
        ws = _FakeWS()
        ws.connect = mock.MagicMock(side_effect=RuntimeError("authentication failed"))
        with self._install_ws(ws):
            out = self.actions["obs_start_recording"]("")
        self.assertIn("rejected the password", out)
        self.assertIn("OBS_PASSWORD", out)

    def test_start_recording_success(self):
        ws = _FakeWS()
        with self._install_ws(ws):
            out = self.actions["obs_start_recording"]("")
        self.assertEqual(out, "Recording, sir.")
        self.assertEqual(ws.calls[0][0], "StartRecord")

    def test_start_recording_already_running_is_noop(self):
        ws = _FakeWS(raise_on={"StartRecord": RuntimeError("output already active")})
        with self._install_ws(ws):
            out = self.actions["obs_start_recording"]("")
        self.assertIn("already recording", out)

    def test_stop_recording_when_not_recording(self):
        ws = _FakeWS(raise_on={"StopRecord": RuntimeError("output not active for record")})
        with self._install_ws(ws):
            out = self.actions["obs_stop_recording"]("")
        self.assertIn("isn't currently recording", out)


class ObsPauseTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("obs_control")
        self.mod._HAS_OBSWS = True

    def _run(self, status_datain):
        ws = _FakeWS(responses={"GetRecordStatus": _Resp(status_datain)})
        with mock.patch.multiple(self.mod, obsws=mock.MagicMock(return_value=ws),
                                 obs_requests=_fake_requests()):
            out = self.actions["obs_pause_recording"]("")
        return out, ws

    def test_pause_when_not_recording(self):
        out, ws = self._run({"outputActive": False, "outputPaused": False})
        self.assertIn("isn't recording", out)
        # Should not have attempted Pause/Resume.
        self.assertNotIn("PauseRecord", [c[0] for c in ws.calls])

    def test_pause_active_recording(self):
        out, ws = self._run({"outputActive": True, "outputPaused": False})
        self.assertIn("paused", out)
        self.assertIn("PauseRecord", [c[0] for c in ws.calls])

    def test_resume_paused_recording(self):
        out, ws = self._run({"outputActive": True, "outputPaused": True})
        self.assertIn("resumed", out)
        self.assertIn("ResumeRecord", [c[0] for c in ws.calls])


class ObsSwitchSceneTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("obs_control")
        self.mod._HAS_OBSWS = True

    def _run(self, arg, scene_names):
        scenes = {"scenes": [{"sceneName": n} for n in scene_names]}
        ws = _FakeWS(responses={"GetSceneList": _Resp(scenes)})
        with mock.patch.multiple(self.mod, obsws=mock.MagicMock(return_value=ws),
                                 obs_requests=_fake_requests()):
            out = self.actions["obs_switch_scene"](arg)
        return out, ws

    def test_requires_name(self):
        self.assertIn("format:", self.actions["obs_switch_scene"](""))

    def test_exact_case_insensitive_match(self):
        out, ws = self._run("gameplay", ["Gameplay", "Starting Soon"])
        self.assertIn("Gameplay", out)
        set_calls = [c for c in ws.calls if c[0] == "SetCurrentProgramScene"]
        self.assertEqual(set_calls[0][1]["sceneName"], "Gameplay")

    def test_unique_substring_match(self):
        out, _ws = self._run("game", ["Gameplay - Main", "Intro"])
        self.assertIn("Gameplay - Main", out)

    def test_ambiguous_substring_refuses(self):
        out, ws = self._run("scene", ["Scene One", "Scene Two"])
        self.assertIn("Multiple scenes match", out)
        self.assertNotIn("SetCurrentProgramScene", [c[0] for c in ws.calls])

    def test_unknown_scene_lists_available(self):
        out, _ws = self._run("nonexistent", ["Alpha", "Beta"])
        self.assertIn("No scene called", out)
        self.assertIn("Alpha", out)


class ObsToggleMuteTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("obs_control")
        self.mod._HAS_OBSWS = True

    def _run(self, arg, input_names, toggle_datain=None):
        inputs = {"inputs": [{"inputName": n} for n in input_names]}
        responses = {"GetInputList": _Resp(inputs),
                     "ToggleInputMute": _Resp(toggle_datain or {})}
        ws = _FakeWS(responses=responses)
        with mock.patch.multiple(self.mod, obsws=mock.MagicMock(return_value=ws),
                                 obs_requests=_fake_requests()):
            out = self.actions["obs_toggle_mute"](arg)
        return out, ws

    def test_requires_source(self):
        self.assertIn("format:", self.actions["obs_toggle_mute"](""))

    def test_resolves_substring_and_reports_muted(self):
        # 'mic' should resolve to 'Mic/Aux'; OBS reports the new state True.
        out, ws = self._run("mic", ["Mic/Aux", "Desktop Audio"],
                            toggle_datain={"inputMuted": True})
        self.assertIn("Mic/Aux", out)
        self.assertIn("muted", out)
        tog = [c for c in ws.calls if c[0] == "ToggleInputMute"]
        self.assertEqual(tog[0][1]["inputName"], "Mic/Aux")

    def test_reports_live_when_unmuted(self):
        out, _ws = self._run("mic", ["Mic/Aux"], toggle_datain={"inputMuted": False})
        self.assertIn("live", out)

    def test_unknown_source_lists_available(self):
        out, _ws = self._run("guitar", ["Mic/Aux", "Desktop Audio"])
        self.assertIn("No audio source called", out)
        self.assertIn("Mic/Aux", out)

    def test_ambiguous_source_refuses(self):
        # Two inputs contain 'aux' → ambiguous, no toggle issued (covers 214).
        out, ws = self._run("aux", ["Mic/Aux", "Game Aux"])
        self.assertIn("Multiple inputs match", out)
        self.assertNotIn("ToggleInputMute", [c[0] for c in ws.calls])

    def test_toggle_unknown_state_generic_confirm(self):
        # OBS returns no inputMuted field → the generic "Toggled mute" confirm
        # (covers 232).
        out, _ws = self._run("mic", ["Mic/Aux"], toggle_datain={})
        self.assertIn("Toggled mute on 'Mic/Aux'", out)

    def test_get_input_list_error(self):
        ws = _FakeWS(raise_on={"GetInputList": RuntimeError("no inputs")})
        with mock.patch.multiple(self.mod, obsws=mock.MagicMock(return_value=ws),
                                 obs_requests=_fake_requests()):
            out = self.actions["obs_toggle_mute"]("mic")
        self.assertIn("didn't return its input list", out)

    def test_toggle_input_mute_error(self):
        inputs = {"inputs": [{"inputName": "Mic/Aux"}]}
        ws = _FakeWS(responses={"GetInputList": _Resp(inputs)},
                     raise_on={"ToggleInputMute": RuntimeError("locked")})
        with mock.patch.multiple(self.mod, obsws=mock.MagicMock(return_value=ws),
                                 obs_requests=_fake_requests()):
            out = self.actions["obs_toggle_mute"]("mic")
        self.assertIn("refused to toggle mute", out)


class ObsErrorBranchTests(unittest.TestCase):
    """Remaining OBS-refusal / API-error branches across the actions."""

    def setUp(self):
        self.mod, self.actions = load_skill_isolated("obs_control")
        self.mod._HAS_OBSWS = True

    def _ws(self, **kw):
        ws = _FakeWS(**kw)
        return ws, mock.patch.multiple(
            self.mod, obsws=mock.MagicMock(return_value=ws),
            obs_requests=_fake_requests())

    def test_start_recording_generic_refusal(self):
        ws, patch = self._ws(raise_on={"StartRecord": RuntimeError("disk error")})
        with patch:
            out = self.actions["obs_start_recording"]("")
        self.assertIn("refused to start recording", out)

    def test_stop_recording_generic_refusal(self):
        ws, patch = self._ws(raise_on={"StopRecord": RuntimeError("disk error")})
        with patch:
            out = self.actions["obs_stop_recording"]("")
        self.assertIn("refused to stop recording", out)

    def test_stop_recording_success(self):
        ws, patch = self._ws()
        with patch:
            out = self.actions["obs_stop_recording"]("")
        self.assertEqual(out, "Recording stopped, sir.")
        self.assertEqual(ws.calls[0][0], "StopRecord")

    def test_pause_get_status_error(self):
        ws, patch = self._ws(raise_on={"GetRecordStatus": RuntimeError("no reply")})
        with patch:
            out = self.actions["obs_pause_recording"]("")
        self.assertIn("didn't answer about recording state", out)

    def test_pause_toggle_error(self):
        ws, patch = self._ws(
            responses={"GetRecordStatus": _Resp({"outputActive": True,
                                                 "outputPaused": False})},
            raise_on={"PauseRecord": RuntimeError("busy")})
        with patch:
            out = self.actions["obs_pause_recording"]("")
        self.assertIn("refused to toggle pause", out)

    def test_switch_scene_get_list_error(self):
        ws, patch = self._ws(raise_on={"GetSceneList": RuntimeError("no scenes")})
        with patch:
            out = self.actions["obs_switch_scene"]("gameplay")
        self.assertIn("didn't return its scene list", out)

    def test_switch_scene_set_error(self):
        ws, patch = self._ws(
            responses={"GetSceneList": _Resp(
                {"scenes": [{"sceneName": "Gameplay"}]})},
            raise_on={"SetCurrentProgramScene": RuntimeError("locked")})
        with patch:
            out = self.actions["obs_switch_scene"]("gameplay")
        self.assertIn("refused to switch scene", out)

    def test_disconnect_error_is_swallowed(self):
        # ws.disconnect() raising in the finally must not break the result
        # (covers the except-pass at 99-100).
        ws = _FakeWS()
        ws.disconnect = mock.MagicMock(side_effect=RuntimeError("already closed"))
        with mock.patch.multiple(self.mod, obsws=mock.MagicMock(return_value=ws),
                                 obs_requests=_fake_requests()):
            out = self.actions["obs_start_recording"]("")
        self.assertEqual(out, "Recording, sir.")


class ObsFailedRequestStatusTests(unittest.TestCase):
    """Regression: obs-websocket-py returns normally on a failed request with
    .status=False — it does NOT raise. The old handlers only caught exceptions,
    so every OBS refusal was reported as success ('Recording, sir.' while
    nothing recorded). These assert the .status check catches the failure."""

    def setUp(self):
        self.mod, self.actions = load_skill_isolated("obs_control")
        self.mod._HAS_OBSWS = True

    def _ws(self, responses):
        ws = _FakeWS(responses=responses)
        return ws, mock.patch.multiple(
            self.mod, obsws=mock.MagicMock(return_value=ws),
            obs_requests=_fake_requests())

    def test_start_failed_already_active_is_friendly_noop(self):
        _ws, patch = self._ws({"StartRecord": _Resp(
            {"comment": "The output is already active."}, status=False)})
        with patch:
            out = self.actions["obs_start_recording"]("")
        self.assertEqual(out, "OBS is already recording, sir.")

    def test_start_failed_generic_reports_comment_not_success(self):
        _ws, patch = self._ws({"StartRecord": _Resp(
            {"comment": "No output configured."}, status=False)})
        with patch:
            out = self.actions["obs_start_recording"]("")
        self.assertIn("refused to start recording", out)
        self.assertIn("No output configured.", out)

    def test_start_failed_without_comment_still_not_success(self):
        _ws, patch = self._ws({"StartRecord": _Resp({}, status=False)})
        with patch:
            out = self.actions["obs_start_recording"]("")
        self.assertNotEqual(out, "Recording, sir.")
        self.assertIn("refused to start recording", out)

    def test_stop_failed_not_active_is_friendly_noop(self):
        _ws, patch = self._ws({"StopRecord": _Resp(
            {"comment": "The output is not active."}, status=False)})
        with patch:
            out = self.actions["obs_stop_recording"]("")
        self.assertEqual(out, "OBS isn't currently recording, sir.")

    def test_pause_toggle_failed_status(self):
        _ws, patch = self._ws({
            "GetRecordStatus": _Resp({"outputActive": True, "outputPaused": False}),
            "PauseRecord": _Resp({"comment": "Pause unsupported."}, status=False)})
        with patch:
            out = self.actions["obs_pause_recording"]("")
        self.assertIn("refused to toggle pause", out)
        self.assertNotIn("paused, sir", out)

    def test_get_record_status_failed_status(self):
        _ws, patch = self._ws({"GetRecordStatus": _Resp(
            {"comment": "Not ready."}, status=False)})
        with patch:
            out = self.actions["obs_pause_recording"]("")
        self.assertIn("didn't answer about recording state", out)

    def test_switch_scene_set_failed_status_not_reported_as_switched(self):
        _ws, patch = self._ws({
            "GetSceneList": _Resp({"scenes": [{"sceneName": "Gameplay"}]}),
            "SetCurrentProgramScene": _Resp({"comment": "ResourceNotFound"},
                                            status=False)})
        with patch:
            out = self.actions["obs_switch_scene"]("gameplay")
        self.assertIn("refused to switch scene", out)
        self.assertNotIn("Scene switched", out)

    def test_get_scene_list_failed_status(self):
        _ws, patch = self._ws({"GetSceneList": _Resp(
            {"comment": "Not ready."}, status=False)})
        with patch:
            out = self.actions["obs_switch_scene"]("gameplay")
        self.assertIn("didn't return its scene list", out)

    def test_toggle_mute_failed_status_not_reported_as_toggled(self):
        _ws, patch = self._ws({
            "GetInputList": _Resp({"inputs": [{"inputName": "Mic/Aux"}]}),
            "ToggleInputMute": _Resp({"comment": "ResourceNotFound"},
                                     status=False)})
        with patch:
            out = self.actions["obs_toggle_mute"]("mic")
        self.assertIn("refused to toggle mute", out)
        self.assertNotIn("muted, sir", out)

    def test_get_input_list_failed_status(self):
        _ws, patch = self._ws({"GetInputList": _Resp(
            {"comment": "Not ready."}, status=False)})
        with patch:
            out = self.actions["obs_toggle_mute"]("mic")
        self.assertIn("didn't return its input list", out)


class ObsImportFallbackTests(unittest.TestCase):
    """The import-time `except` that degrades gracefully when
    obs-websocket-py is missing (lines 48-52). We re-exec the module with the
    library blocked in sys.modules so the import genuinely fails."""

    def test_missing_library_sets_flags(self):
        import importlib.util
        import sys

        path = self.mod_path()
        # Block the library so `from obswebsocket import ...` raises ImportError.
        blocked = {"obswebsocket": None}
        with mock.patch.dict(sys.modules, blocked):
            spec = importlib.util.spec_from_file_location("obs_control_reimport", path)
            fresh = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(fresh)
        self.assertFalse(fresh._HAS_OBSWS)
        self.assertIsNotNone(fresh._IMPORT_ERROR)
        self.assertIsNone(fresh.obsws)
        # And the missing-dep hint surfaces the captured import error.
        self.assertIn("obs-websocket-py is not installed", fresh._missing_dep_msg())

    def mod_path(self):
        from tests._skill_harness import skill_path
        p, _ = skill_path("obs_control")
        return p


if __name__ == "__main__":
    unittest.main()
