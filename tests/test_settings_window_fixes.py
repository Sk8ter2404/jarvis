"""Tests for the 2026-09-30 Settings-window fixes (tools/settings_window.py).

NO WINDOW IS EVER OPENED. The GUI is driven through a FAKE tkinter (the
FakeTk harness below): widgets are plain Python objects that record what they
were given, so the real window code — run_gui(), SettingsApp, the old run_gui
too — executes end to end with no display, no Tk and no popups. Nothing here
touches data/user_settings.json (every test uses a temp file), opens an audio
stream (device lists are fixtures), reaches Ollama or the web interface (the
probes are replaced), or signals JARVIS (the tray inbox is a temp file).

Several tests drive only behaviour the OLD code also had an entry point for
(run_gui, save_settings, load_settings, SCHEMA), so they fail on the pre-fix
tree for the right reason, not just because a new name is missing.

PRIVACY: fixture values only — device names are hardware model strings.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest import mock

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)
_TOOLS = os.path.join(_PROJECT, "tools")
_MODULE_PATH = os.path.join(_TOOLS, "settings_window.py")

_spec = importlib.util.spec_from_file_location("jarvis_settings_window_fixes",
                                               _MODULE_PATH)
sw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sw)

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0) \
    if sys.platform == "win32" else 0

# This desk's real PortAudio list (names only; nothing is opened), as
# sounddevice.query_devices() returns it — the fixture for the device picker.
_HOSTAPIS = [{"name": "MME"}, {"name": "Windows DirectSound"},
             {"name": "Windows WASAPI"}, {"name": "Windows WDM-KS"}]


def _dev(name, api, ins, outs):
    return {"name": name, "hostapi": api, "max_input_channels": ins,
            "max_output_channels": outs}


_DEVICES = [
    _dev("Microsoft Sound Mapper - Input", 0, 2, 0),
    _dev("Microphone (Blue Snowball )", 0, 1, 0),
    _dev("Microsoft Sound Mapper - Output", 0, 0, 2),
    _dev("Speakers (Realtek USB2.0 Audio)", 0, 0, 8),
    _dev("U28E510 (2- NVIDIA High Definit", 0, 0, 2),
    _dev("Primary Sound Capture Driver", 1, 2, 0),
    _dev("Microphone (Blue Snowball )", 1, 1, 0),
    _dev("Primary Sound Driver", 1, 0, 2),
    _dev("Speakers (Realtek USB2.0 Audio)", 1, 0, 8),
    _dev("U28E510 (2- NVIDIA High Definition Audio)", 1, 0, 2),
    _dev("U28E510 (2- NVIDIA High Definition Audio)", 2, 0, 2),
    _dev("Speakers (Realtek USB2.0 Audio)", 2, 0, 2),
    _dev("Microphone (Blue Snowball )", 2, 2, 0),
    _dev("Microphone ()", 3, 2, 0),
    _dev("Microphone Array (Xbox NUI Sensor)", 3, 4, 0),
    _dev("Line (Realtek USB2.0 Audio)", 3, 2, 0),
    _dev("Analog Connector (Realtek USB2.0 Audio)", 3, 2, 0),
    _dev("Microphone (Blue Snowball)", 3, 1, 0),
    _dev("Microphone (Steam Streaming Microphone Wave)", 3, 8, 0),
    _dev("Input (Steam Streaming Speakers Wave)", 3, 8, 0),
    _dev("Output ()", 3, 0, 2),
]


def _pick_device(preferred, devices, want_input=True):
    """bobert_companion._pick_device's matching rule (case-insensitive
    substring, preferred order, then index order), minus the open check."""
    for pref in preferred:
        for i, d in enumerate(devices):
            if pref.lower() not in d["name"].lower():
                continue
            if want_input and d["max_input_channels"] > 0:
                return i, d["name"]
            if not want_input and d["max_output_channels"] > 0:
                return i, d["name"]
    return None, ""


# ════════════════════════════════════════════════════════════════════════
#  The fake tkinter
# ════════════════════════════════════════════════════════════════════════
class FakeTk:
    """A stand-in for the tkinter / ttk / messagebox modules. Everything a
    widget is given is recorded; mainloop() runs ``on_mainloop(root)`` once
    (the "user") and returns."""

    def __init__(self):
        h = self
        self.widgets: list = []
        self.vars: list = []
        self.styles: list = []
        self.roots: list = []
        self.focus = None
        self.on_mainloop = None
        self.ask_answer = False
        self.asked: list = []
        self.shown_errors: list = []

        class Var:
            def __init__(self, master=None, value=None, name=None):
                self._v = value
                self._traces = []
                h.vars.append(self)

            def get(self):
                return self._v

            def set(self, v):
                self._v = v
                for cb in list(self._traces):
                    cb("PY_VAR", "", "write")

            def trace_add(self, mode, cb):
                self._traces.append(cb)
                return "trace"

        class BooleanVar(Var):
            def get(self):
                return bool(self._v)

        class Widget:
            def __init__(self, *args, **kw):
                self.master = args[0] if args else None
                self.kw = dict(kw)
                self.bindings = {}
                self.visible = None
                self.pack_kw = None
                h.widgets.append(self)

            def configure(self, cnf=None, **kw):
                if cnf:
                    kw.update(cnf)
                self.kw.update(kw)

            config = configure

            def cget(self, key):
                return self.kw.get(key)

            def __getitem__(self, key):
                return self.kw.get(key)

            def __setitem__(self, key, value):
                self.kw[key] = value

            def bind(self, seq=None, func=None, add=None):
                self.bindings[seq] = func

            def bind_all(self, seq=None, func=None, add=None):
                h.bound_all = (seq, func)

            def unbind_all(self, seq):
                pass

            def grid(self, **kw):
                self.visible = True

            def grid_remove(self):
                self.visible = False

            def pack(self, **kw):
                self.pack_kw = kw

            def pack_configure(self, **kw):
                pass

            def columnconfigure(self, *a, **kw):
                pass

            def winfo_width(self):
                return 700

            def focus_get(self):
                return h.focus

            def event_generate(self, *a, **kw):
                pass

            def __str__(self):
                return f".fake{id(self)}"

            def __getattr__(self, name):
                if name.startswith("_"):
                    raise AttributeError(name)
                return lambda *a, **k: None

        class Text(Widget):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                self.content = ""

            def insert(self, index, s):
                self.content = s + self.content if index == "1.0" \
                    else self.content + s

            def get(self, start, end=None):
                return self.content + ("" if end == "end-1c" else "\n")

            def delete(self, start, end=None):
                self.content = ""

        class Canvas(Widget):
            def create_window(self, *a, **kw):
                return 1

            def create_rectangle(self, *a, **kw):
                return 2

            def bbox(self, *a):
                return (0, 0, 10, 10)

            def yview(self, *a):
                return (0.0, 1.0)

            def yview_scroll(self, n, what):
                self.kw.setdefault("_scrolled", []).append(n)

        class Combobox(Widget):
            def current(self, index=None):
                values = list(self.kw.get("values") or [])
                var = self.kw.get("textvariable")
                if index is None:
                    v = var.get() if var is not None else None
                    return values.index(v) if v in values else -1
                var.set(values[index])

        class Notebook(Widget):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                self.tabs = []
                self.selected = None

            def add(self, child, **kw):
                self.tabs.append(kw.get("text"))

            def select(self, tab_id=None):
                self.selected = tab_id

            def index(self, what):
                return self.selected if isinstance(self.selected, int) else 0

        class Tk(Widget):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                self.afters = []
                self.attrs = []
                self.options = {}
                self.protocols = {}
                self.destroyed = False
                self.lifted = 0
                h.roots.append(self)

            def title(self, t=None):
                self.kw["title"] = t

            def geometry(self, g=None):
                self.kw["geometry"] = g

            def minsize(self, *a):
                pass

            def attributes(self, *a):
                self.attrs.append(a)

            def after(self, ms, fn=None, *args):
                self.afters.append((ms, fn))
                return f"after#{len(self.afters)}"

            def after_cancel(self, _id):
                pass

            def option_add(self, pattern, value, priority=None):
                self.options[pattern] = value

            def protocol(self, name, fn=None):
                self.protocols[name] = fn

            def winfo_fpixels(self, s):
                return 96.0

            def lift(self, *a):
                self.lifted += 1

            def deiconify(self):
                pass

            def focus_force(self):
                pass

            def destroy(self):
                self.destroyed = True

            def mainloop(self, n=0):
                if h.on_mainloop is not None:
                    h.on_mainloop(self)

            def run_afters(self):
                pending, self.afters = list(self.afters), []
                for _ms, fn in pending:
                    if fn is not None:
                        fn()

        class Style:
            def __init__(self, *a, **kw):
                self.configured = {}
                self.maps = {}
                self.theme = None
                h.styles.append(self)

            def theme_use(self, name=None):
                self.theme = name

            def configure(self, style, query_opt=None, **kw):
                self.configured.setdefault(style, {}).update(kw)

            def map(self, style, query_opt=None, **kw):
                self.maps.setdefault(style, {}).update(kw)

            def lookup(self, *a, **kw):
                return ""

        def showerror(title=None, message=None, **kw):
            h.shown_errors.append(message)

        def askyesnocancel(title=None, message=None, **kw):
            h.asked.append(message)
            return h.ask_answer

        self.Tk_cls = Tk
        self.tk = types.SimpleNamespace(
            Tk=Tk, StringVar=Var, BooleanVar=BooleanVar, IntVar=Var,
            Canvas=Canvas, Label=Widget, Entry=Widget, Text=Text,
            Frame=Widget, Toplevel=Tk)
        self.ttk = types.SimpleNamespace(
            Style=Style, Notebook=Notebook, Frame=Widget, Label=Widget,
            Checkbutton=Widget, Combobox=Combobox, Button=Widget,
            Scrollbar=Widget, Separator=Widget, Entry=Widget)
        self.messagebox = types.SimpleNamespace(
            showerror=showerror, askyesnocancel=askyesnocancel,
            showinfo=lambda *a, **k: None, showwarning=lambda *a, **k: None)

    def sys_modules(self) -> dict:
        tkmod = types.ModuleType("tkinter")
        tkmod.__dict__.update(vars(self.tk))
        ttkmod = types.ModuleType("tkinter.ttk")
        ttkmod.__dict__.update(vars(self.ttk))
        mbmod = types.ModuleType("tkinter.messagebox")
        mbmod.__dict__.update(vars(self.messagebox))
        tkmod.ttk = ttkmod
        tkmod.messagebox = mbmod
        return {"tkinter": tkmod, "tkinter.ttk": ttkmod,
                "tkinter.messagebox": mbmod}

    # finders
    def button(self, text):
        for w in self.widgets:
            if w.kw.get("text") == text and "command" in w.kw:
                return w
        return None

    def comboboxes(self):
        return [w for w in self.widgets if type(w).__name__ == "Combobox"]


def _patch_if_present(stack, obj, name, value):
    if hasattr(obj, name):
        stack.enter_context(mock.patch.object(obj, name, value))


def _run_gui_with_fake_tk(fake: FakeTk, settings_path: str,
                          command_path: str | None = None):
    """Run the module's real run_gui() against the fake tkinter, with every
    side-effecting probe replaced. Works for the old run_gui and the new one."""
    with contextlib.ExitStack() as st:
        st.enter_context(mock.patch.dict(sys.modules, fake.sys_modules()))
        st.enter_context(mock.patch.dict(
            os.environ, {"JARVIS_SETTINGS_PATH": settings_path}))
        _patch_if_present(st, sw, "installed_ollama_models",
                          lambda *a, **k: ["gemma4:26b-a4b-it-qat"])
        _patch_if_present(st, sw, "list_input_devices", lambda: [])
        _patch_if_present(st, sw, "_query_audio_devices",
                          lambda: (list(_DEVICES), list(_HOSTAPIS)))
        _patch_if_present(st, sw, "fetch_pending_restart", lambda *a, **k: None)
        _patch_if_present(st, sw, "_load_vram_budget", lambda: None)
        if command_path:
            _patch_if_present(st, sw, "TRAY_COMMANDS_FILE", command_path)
        with contextlib.redirect_stderr(io.StringIO()):
            return sw.run_gui(0)


class _TmpDir(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        self.dir = self._td.name
        self.path = os.path.join(self.dir, "user_settings.json")
        self.cmd = os.path.join(self.dir, "tray_commands.json")

    def write(self, doc):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(doc, f)

    def read(self):
        with open(self.path, encoding="utf-8") as f:
            return json.load(f)


def _config_lookup(overrides=None):
    table = {"MONITORS": {"top": (0, 0, 1, 1), "left": (0, 0, 1, 1)},
             "CAMERAS": [{"index": 0, "label": "Desk cam", "name": "usb cam",
                          "primary": True}]}
    table.update(overrides or {})
    return lambda key: table.get(key, sw._MISSING)


def _wait_async(app, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        if all(s.get("done") for s in app._async.values()):
            return
        time.sleep(0.01)


def _app(test, fake=None, **kw):
    fake = fake or FakeTk()
    kw.setdefault("path", test.path)
    kw.setdefault("config_lookup", _config_lookup())
    kw.setdefault("audio_devices", (list(_DEVICES), list(_HOSTAPIS)))
    kw.setdefault("model_probe", lambda include_vision=False: (
        ["gemma4:26b-a4b-it-qat", "gemma4:12b"]
        + (["qwen2.5vl:7b"] if include_vision else [])))
    kw.setdefault("running_probe", lambda values: None)
    kw.setdefault("find_spec", lambda name: object())
    kw.setdefault("command_path", test.cmd)
    kw.setdefault("file_opener", lambda p: None)
    kw.setdefault("total_vram_mb", 24576)
    app = sw.SettingsApp(fake.tk, fake.ttk, fake.messagebox, **kw)
    _wait_async(app)
    return fake, app


# ════════════════════════════════════════════════════════════════════════
#  P0-1  opened from the tray, the core imports must resolve
# ════════════════════════════════════════════════════════════════════════
class TrayLaunchImportTests(unittest.TestCase):
    """tray.py runs ``pythonw tools\\settings_window.py`` — a SCRIPT, so
    sys.path[0] is tools\\ and the project root is not importable unless the
    file puts it back. Run it that way, in a subprocess with no PYTHONPATH."""

    def _env(self, d):
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        env["JARVIS_SETTINGS_PATH"] = os.path.join(d, "never_written.json")
        return env

    def test_script_context_import_resolves_core(self):
        probe = ("import json, sys; sys.path[0] = sys.argv[1]; "
                 "import settings_window as sw; "
                 "print(json.dumps({'vram': sw._load_vram_budget() is not None,"
                 " 'lockstep': sw._model_lockstep() is not None}))")
        with tempfile.TemporaryDirectory() as d:
            r = subprocess.run([sys.executable, "-c", probe, _TOOLS], cwd=d,
                               env=self._env(d), capture_output=True,
                               text=True, timeout=120,
                               creationflags=_NO_WINDOW)
        self.assertEqual(r.returncode, 0, r.stderr)
        got = json.loads(r.stdout.strip().splitlines()[-1])
        self.assertEqual(got, {"vram": True, "lockstep": True},
                         "run as the tray runs it, settings_window can't "
                         "import core — the VRAM panel reads 'unavailable' "
                         "and the vision lockstep is skipped")

    def test_selftest_flag_runs_the_script_without_a_window(self):
        with tempfile.TemporaryDirectory() as d:
            r = subprocess.run([sys.executable, _MODULE_PATH, "--selftest"],
                               cwd=d, env=self._env(d), capture_output=True,
                               text=True, timeout=120,
                               creationflags=_NO_WINDOW)
            self.assertEqual(os.listdir(d), [], "--selftest wrote a file")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        report = json.loads(r.stdout)
        self.assertTrue(report["ok"])
        self.assertEqual(report["imports"], {"core.model_lockstep": "ok",
                                             "core.vram_budget": "ok"})
        self.assertTrue(report["vram_panel"])
        self.assertTrue(report["vision_lockstep"])


# ════════════════════════════════════════════════════════════════════════
#  P0-2  read-only dropdowns must be readable
# ════════════════════════════════════════════════════════════════════════
class DarkReadonlyComboboxTests(_TmpDir):
    def _assert_dark(self, style, root):
        cmap = style.maps.get("TCombobox", {})
        self.assertIn(("readonly", sw.FIELD_BG),
                      list(cmap.get("fieldbackground", [])),
                      "clam paints a READONLY combobox field #dcdad5 unless "
                      "the readonly state is mapped — FG text vanishes on it")
        self.assertIn(("readonly", sw.FG), list(cmap.get("foreground", [])))
        self.assertEqual(root.options.get("*TCombobox*Listbox.background"),
                         sw.FIELD_BG, "the dropdown list stays white")
        self.assertEqual(root.options.get("*TCombobox*Listbox.foreground"),
                         sw.FG)

    def test_run_gui_maps_the_readonly_state_and_the_list(self):
        fake = FakeTk()
        _run_gui_with_fake_tk(fake, self.path)
        self.assertTrue(fake.styles and fake.roots)
        self._assert_dark(fake.styles[0], fake.roots[0])

    def test_apply_dark_theme_unit(self):
        fake = FakeTk()
        style, root = fake.ttk.Style(), fake.tk.Tk()
        sw.apply_dark_theme(style, root)
        self._assert_dark(style, root)
        self.assertEqual(style.theme, "clam")


# ════════════════════════════════════════════════════════════════════════
#  P0-3  the hand-mouse row must drive the LIVE hand-mouse
# ════════════════════════════════════════════════════════════════════════
class HandMouseRowTests(unittest.TestCase):
    def test_hand_mouse_row_is_the_live_knob(self):
        spec = sw.SCHEMA.get("KINECT_AIR_MOUSE_ENABLED")
        self.assertIsNotNone(spec, "no row for the live hand-mouse knob")
        self.assertEqual(spec["type"], "bool")
        self.assertIs(spec["default"], False)
        self.assertIn("hand-mouse", spec["label"].lower())
        self.assertIn("KINECT_AIR_MOUSE_ENABLED", sw.persisted_keys())

    def test_air_control_row_no_longer_claims_to_be_the_hand_mouse(self):
        label = sw.SCHEMA["AIR_CONTROL_ENABLED"]["label"].lower()
        help_ = sw.SCHEMA["AIR_CONTROL_ENABLED"]["help"].lower()
        self.assertNotIn("hand-mouse", label)
        self.assertIn("dormant", label)
        self.assertIn("not the air-mouse", help_)
        # Only ONE row may present itself as the hand-mouse switch.
        claim = [k for k, s in sw.SCHEMA.items()
                 if "hand-mouse" in str(s.get("label", "")).lower()]
        self.assertEqual(claim, ["KINECT_AIR_MOUSE_ENABLED"])

    def test_both_engines_on_is_flagged(self):
        notes = sw.effective_warnings({"AIR_CONTROL_ENABLED": True,
                                       "KINECT_AIR_MOUSE_ENABLED": True},
                                      find_spec=lambda m: object())
        self.assertIn("AIR_CONTROL_ENABLED", notes)


# ════════════════════════════════════════════════════════════════════════
#  P0-5  a file that can't be parsed must never be saved over
# ════════════════════════════════════════════════════════════════════════
_OWNER_DOC = {"AI_BACKEND": "ollama", "TTS_BACKEND": "kokoro",
              "CAMERAS": [{"index": 2, "name": "fake cam"}],
              "KINECT_GUARD_ENABLED": True, "LOCAL_VISION_MODEL": "gemma4:12b",
              "REQUIRE_WAKE_MODE": False}


class UnreadableFileTests(_TmpDir):
    def _write_bom(self):
        with open(self.path, "wb") as f:
            f.write(b"\xef\xbb\xbf" + json.dumps(_OWNER_DOC).encode("utf-8"))

    def _write_trailing_comma(self):
        text = json.dumps(_OWNER_DOC, indent=2)
        with open(self.path, "w", encoding="utf-8") as f:
            f.write(text[:-2] + ",\n}\n")

    def test_bom_file_loads_its_values(self):
        self._write_bom()
        s = sw.load_settings(self.path)
        self.assertEqual(s["AI_BACKEND"], "ollama")
        self.assertEqual(s["TTS_BACKEND"], "kokoro")
        self.assertIn("CAMERAS", s)

    def test_bom_file_survives_a_voice_toggle(self):
        self._write_bom()
        cur = sw.load_settings(self.path)
        cur["REQUIRE_WAKE_MODE"] = True
        sw.save_settings(cur, self.path)
        doc = self.read()
        self.assertIs(doc["REQUIRE_WAKE_MODE"], True)
        self.assertEqual(doc["CAMERAS"], _OWNER_DOC["CAMERAS"])
        self.assertEqual(doc["AI_BACKEND"], "ollama")

    def test_save_refuses_to_overwrite_an_unparseable_file(self):
        self._write_trailing_comma()
        with open(self.path, "rb") as f:
            before = f.read()
        # The voice toggles' load → change → save pattern.
        cur = sw.load_settings(self.path)
        cur["REQUIRE_WAKE_MODE"] = True
        with self.assertRaises(ValueError):
            sw.save_settings(cur, self.path)
        with open(self.path, "rb") as f:
            self.assertEqual(f.read(), before,
                             "a save wrote over a file it could not read — "
                             "CAMERAS, KINECT_* and every other key it "
                             "didn't manage are gone")

    def test_non_object_document_is_refused_too(self):
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("[1, 2, 3]")
        with self.assertRaises(ValueError):
            sw.save_settings(sw.load_settings(self.path), self.path)

    def test_error_names_line_and_column(self):
        self._write_trailing_comma()
        with self.assertRaises(sw.SettingsFileError) as cm:
            sw.read_settings_file(self.path)
        self.assertRegex(str(cm.exception), r"line \d+, column \d+")
        self.assertIsNone(sw.settings_file_problem(
            os.path.join(self.dir, "absent.json")))

    def test_update_and_save_changed_refuse_too(self):
        self._write_trailing_comma()
        with self.assertRaises(sw.SettingsFileError):
            sw.update_settings({"TTS_VOICE": "x"}, self.path)
        with self.assertRaises(sw.SettingsFileError):
            sw.save_changed_settings({"TTS_VOICE": "x"}, self.path)

    def test_gui_save_over_an_unparseable_file_is_refused(self):
        # Through the real run_gui (old or new): press Save with no edits.
        self._write_trailing_comma()
        with open(self.path, "rb") as f:
            before = f.read()
        fake = FakeTk()

        def user(root):
            fake.button("Save").kw["command"]()
        fake.on_mainloop = user
        _run_gui_with_fake_tk(fake, self.path)
        with open(self.path, "rb") as f:
            self.assertEqual(f.read(), before)

    def test_gui_shows_the_problem_and_disables_save(self):
        self._write_trailing_comma()
        fake, app = _app(self)
        self.assertTrue(app.file_error)
        self.assertIn("line", app.banner.kw.get("text", ""))
        self.assertEqual(app.buttons["save"].kw.get("state"), "disabled")
        self.assertEqual(app.buttons["restart"].kw.get("state"), "disabled")
        self.assertFalse(app.save())


# ════════════════════════════════════════════════════════════════════════
#  P0-6 / P1-2  Save writes only what the owner changed
# ════════════════════════════════════════════════════════════════════════
class ChangedFieldsOnlyTests(_TmpDir):
    def test_gui_save_keeps_a_voice_change_made_while_open(self):
        self.write(dict(_OWNER_DOC))
        fake = FakeTk()

        def user(root):
            # A voice command flips wake-word mode while the window is open…
            cur = sw.load_settings(self.path)
            cur["REQUIRE_WAKE_MODE"] = True
            sw.save_settings(cur, self.path)
            # …then the owner presses Save without touching that row.
            fake.button("Save").kw["command"]()
        fake.on_mainloop = user
        _run_gui_with_fake_tk(fake, self.path)
        self.assertIs(self.read()["REQUIRE_WAKE_MODE"], True,
                      "Save reverted a setting changed by voice while the "
                      "window was open")

    def test_gui_save_does_not_freeze_untouched_defaults(self):
        self.write({"AI_BACKEND": "ollama"})
        fake = FakeTk()
        fake.on_mainloop = lambda root: fake.button("Save").kw["command"]()
        _run_gui_with_fake_tk(fake, self.path)
        doc = self.read()
        self.assertNotIn("TTS_VOICE", doc,
                         "a Save with no edits pinned every default into the "
                         "file")
        self.assertEqual(doc, {"AI_BACKEND": "ollama"})

    def test_only_the_edited_field_is_written(self):
        self.write(dict(_OWNER_DOC))
        fake, app = _app(self)
        cur = sw.load_settings(self.path)          # voice change meanwhile
        cur["REQUIRE_WAKE_MODE"] = True
        sw.save_settings(cur, self.path)
        app.fields["VAD_THRESHOLD"].set_raw("0.01")
        self.assertTrue(app.save())
        doc = self.read()
        self.assertEqual(doc["VAD_THRESHOLD"], 0.01)
        self.assertIs(doc["REQUIRE_WAKE_MODE"], True)
        self.assertEqual(doc["CAMERAS"], _OWNER_DOC["CAMERAS"])
        self.assertIs(doc["KINECT_GUARD_ENABLED"], True)
        self.assertEqual(set(doc) - set(_OWNER_DOC), {"VAD_THRESHOLD"})
        # A second Save has nothing to do.
        with open(self.path, "rb") as f:
            before = f.read()
        self.assertTrue(app.save())
        with open(self.path, "rb") as f:
            self.assertEqual(f.read(), before)

    def test_typing_back_the_original_value_is_not_a_change(self):
        self.write({"VAD_THRESHOLD": 0.008})
        fake, app = _app(self)
        app.fields["VAD_THRESHOLD"].set_raw("0.0080")
        self.assertEqual(app.collect(), ({}, {}))

    def test_save_changed_settings_merges_into_disk(self):
        self.write({"CAMERAS": [1], "TTS_VOICE": "a"})
        doc, lock = sw.save_changed_settings({"TTS_VOICE": "b"}, self.path)
        self.assertEqual(self.read(), {"CAMERAS": [1], "TTS_VOICE": "b"})
        self.assertEqual(lock, (None, ""))

    def test_chat_model_change_carries_vision_unless_vision_also_set(self):
        self.write({"LOCAL_LLM_MODEL": "gemma4:26b-a4b-it-qat",
                    "LOCAL_VISION_MODEL": "gemma4:26b-a4b-it-qat"})
        _doc, (tag, _r) = sw.save_changed_settings(
            {"LOCAL_LLM_MODEL": "gemma4:12b"}, self.path)
        self.assertEqual(tag, "gemma4:12b")
        self.assertEqual(self.read()["LOCAL_VISION_MODEL"], "gemma4:12b")
        # An explicit vision choice in the same save wins.
        sw.save_changed_settings({"LOCAL_LLM_MODEL": "gemma4:26b-a4b-it-qat",
                                  "LOCAL_VISION_MODEL": "off"}, self.path)
        self.assertEqual(self.read()["LOCAL_VISION_MODEL"], "off")

    def test_gui_chat_model_change_moves_the_vision_widget_too(self):
        self.write({"LOCAL_LLM_MODEL": "gemma4:26b-a4b-it-qat",
                    "LOCAL_VISION_MODEL": "gemma4:26b-a4b-it-qat"})
        fake, app = _app(self)
        app.fields["LOCAL_LLM_MODEL"].set_raw("gemma4:12b")
        self.assertTrue(app.save())
        self.assertEqual(app.fields["LOCAL_VISION_MODEL"].get_raw(),
                         "gemma4:12b")
        self.assertIn("Vision model moved", app.status_var.get())
        self.assertEqual(app.collect(), ({}, {}))

    def test_absent_keys_show_the_config_value_not_the_schema_default(self):
        # AUDIO_AUTOSWITCH_ENABLED's config default comes from an env var.
        self.write({})
        fake, app = _app(self, config_lookup=_config_lookup(
            {"AUDIO_AUTOSWITCH_ENABLED": True}))
        self.assertIs(app.fields["AUDIO_AUTOSWITCH_ENABLED"].get_raw(), True)
        self.assertEqual(app.collect(), ({}, {}))


# ════════════════════════════════════════════════════════════════════════
#  P1-1  honest restart story + Save & restart
# ════════════════════════════════════════════════════════════════════════
class RestartTests(_TmpDir):
    def test_run_gui_offers_save_and_restart(self):
        fake = FakeTk()
        _run_gui_with_fake_tk(fake, self.path, command_path=self.cmd)
        self.assertIsNotNone(fake.button("Save & restart JARVIS"))

    def test_restart_note_says_next_start(self):
        self.assertIn("next time JARVIS starts", sw.RESTART_NOTE)

    def test_save_and_restart_writes_the_tray_command(self):
        self.write({})
        fake, app = _app(self)
        app.fields["TTS_VOICE"].set_raw("en-GB-ThomasNeural")
        fake.button("Save & restart JARVIS").kw["command"]()
        self.assertEqual(self.read(), {"TTS_VOICE": "en-GB-ThomasNeural"})
        with open(self.cmd, encoding="utf-8") as f:
            cmds = json.load(f)
        self.assertEqual([c["cmd"] for c in cmds], ["restart"])
        self.assertIn("Restart requested", app.status_var.get())

    def test_restart_is_not_sent_when_the_save_fails(self):
        self.write({})
        fake, app = _app(self)
        app.fields["VAD_THRESHOLD"].set_raw("nan")
        self.assertFalse(app.save(restart=True))
        self.assertFalse(os.path.exists(self.cmd))

    def test_send_tray_command_appends_atomically(self):
        with open(self.cmd, "w", encoding="utf-8") as f:
            json.dump([{"cmd": "open_hud", "ts": 1.0}], f)
        self.assertTrue(sw.send_tray_command("restart", path=self.cmd))
        with open(self.cmd, encoding="utf-8") as f:
            cmds = json.load(f)
        self.assertEqual([c["cmd"] for c in cmds], ["open_hud", "restart"])
        self.assertIsInstance(cmds[-1]["ts"], float)
        self.assertEqual([n for n in os.listdir(self.dir)
                          if n.endswith(".tmp")], [])

    def test_saved_rows_are_badged_until_restart(self):
        self.write({})
        fake, app = _app(self)
        app.fields["TTS_VOICE"].set_raw("en-GB-ThomasNeural")
        app.save()
        self.assertEqual(app.fields["TTS_VOICE"].badge_label.kw["text"],
                         "saved · restart to apply")

    def test_rows_the_running_jarvis_lags_are_badged(self):
        self.write({"TTS_VOICE": "en-GB-ThomasNeural"})
        fake, app = _app(self, running_probe=lambda values: {"TTS_VOICE"})
        app._apply_async()
        self.assertEqual(app.fields["TTS_VOICE"].badge_label.kw["text"],
                         "saved · not running yet")

    def test_fetch_pending_restart_reads_the_web_status(self):
        payload = {"settings": [{"name": "TTS_VOICE", "pending_restart": True},
                                {"name": "VAD_THRESHOLD",
                                 "pending_restart": False}]}
        seen = {}

        class _Resp(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def opener(req, timeout=None):
            seen["url"] = req.full_url
            seen["token"] = req.get_header("X-auth-token")
            return _Resp(json.dumps(payload).encode("utf-8"))
        got = sw.fetch_pending_restart(
            {"WEB_INTERFACE_ENABLED": True, "WEB_INTERFACE_PORT": 8766,
             "WEB_INTERFACE_BIND": "0.0.0.0",
             "WEB_INTERFACE_TOKEN": "fake-token"}, opener=opener)
        self.assertEqual(got, {"TTS_VOICE"})
        self.assertEqual(seen["url"], "http://127.0.0.1:8766/api/settings")
        self.assertEqual(seen["token"], "fake-token")
        self.assertIsNone(sw.fetch_pending_restart(
            {"WEB_INTERFACE_ENABLED": False}, opener=opener))


# ════════════════════════════════════════════════════════════════════════
#  P1-3  validate on Save, with the error on the field
# ════════════════════════════════════════════════════════════════════════
class ValidationTests(_TmpDir):
    def test_bad_values_are_refused_with_a_reason(self):
        cases = [("VAD_THRESHOLD", "nan"), ("VAD_THRESHOLD", "-4"),
                 ("VAD_THRESHOLD", "0"), ("VAD_THRESHOLD", "0,01"),
                 ("WEB_INTERFACE_PORT", "99999"), ("WEB_INTERFACE_PORT", "0"),
                 ("WEB_INTERFACE_PORT", "8443"),
                 ("PROCESSING_FILLER_DELAY", "0.1"),
                 ("SELF_ECHO_WINDOW_S", "20s"), ("SELF_ECHO_WINDOW_S", "inf"),
                 ("AUDIO_FLAP_THRESHOLD", "2.5"), ("LOCAL_LLM_MODEL", "  "),
                 ("TTS_VOICE", ""), ("TTS_BACKEND", "nope")]
        for key, raw in cases:
            value, err = sw.validate_value(sw.SCHEMA[key], raw)
            self.assertIsNone(value, (key, raw))
            self.assertTrue(err, (key, raw))

    def test_good_values_parse(self):
        cases = [("AUDIO_FLAP_THRESHOLD", "3.0", 3),
                 ("CAMERA_CULPRIT_THRESHOLD", "5 ", 5),
                 ("VAD_THRESHOLD", "0.01", 0.01),
                 ("WEB_INTERFACE_PORT", "8766", 8766),
                 ("TTS_VOICE", " en-GB-RyanNeural ", "en-GB-RyanNeural"),
                 ("FOLLOWUP_WINDOW_S", "0", 0.0)]
        for key, raw, want in cases:
            self.assertEqual(sw.validate_value(sw.SCHEMA[key], raw),
                             (want, None), (key, raw))

    def test_every_numeric_default_passes_its_own_range(self):
        for key in sw.persisted_keys():
            spec = sw.SCHEMA[key]
            if spec["type"] in ("int", "float"):
                self.assertEqual(sw.validate_value(spec, spec["default"]),
                                 (spec["default"], None), key)

    def test_gui_refuses_the_whole_save_and_marks_the_field(self):
        self.write({"TTS_VOICE": "a"})
        fake, app = _app(self)
        app.fields["VAD_THRESHOLD"].set_raw("-4")
        app.fields["TTS_VOICE"].set_raw("b")
        self.assertFalse(app.save())
        self.assertEqual(self.read(), {"TTS_VOICE": "a"})
        err = app.fields["VAD_THRESHOLD"].error_label
        self.assertTrue(err.visible)
        self.assertIn("more than 0", err.kw["text"])
        self.assertIn("fix the 1 field", app.status_var.get())
        # Fix it: the error clears and both fields save.
        app.fields["VAD_THRESHOLD"].set_raw("0.02")
        self.assertFalse(err.visible)
        self.assertTrue(app.save())
        self.assertEqual(self.read(), {"TTS_VOICE": "b",
                                       "VAD_THRESHOLD": 0.02})

    def test_a_bad_value_already_in_the_file_does_not_block_other_saves(self):
        self.write({"VAD_THRESHOLD": "nan"})
        fake, app = _app(self)
        app.fields["TTS_VOICE"].set_raw("en-GB-ThomasNeural")
        self.assertTrue(app.save())
        self.assertEqual(self.read()["VAD_THRESHOLD"], "nan")


# ════════════════════════════════════════════════════════════════════════
#  P1-4  the wheel over a dropdown scrolls the page
# ════════════════════════════════════════════════════════════════════════
class WheelTests(_TmpDir):
    def test_run_gui_comboboxes_guard_the_wheel(self):
        self.write({})
        fake = FakeTk()
        _run_gui_with_fake_tk(fake, self.path)
        combos = fake.comboboxes()
        self.assertTrue(combos)
        unguarded = [c for c in combos if "<MouseWheel>" not in c.bindings]
        self.assertEqual(unguarded, [],
                         "a combobox without its own wheel binding changes "
                         "value when the page is scrolled over it")

    def _combo_event(self, fake, combo, delta=-120):
        return types.SimpleNamespace(widget=combo, delta=delta)

    def test_unfocused_combobox_keeps_its_value_and_the_page_scrolls(self):
        self.write({})
        fake, app = _app(self)
        combo = app.fields["TTS_BACKEND"].combo
        before = app.fields["TTS_BACKEND"].get_raw()
        fake.focus = None
        handler = combo.bindings["<MouseWheel>"]
        self.assertEqual(handler(self._combo_event(fake, combo)), "break")
        self.assertEqual(app.fields["TTS_BACKEND"].get_raw(), before)
        scrolled = [w for w in fake.widgets
                    if w.kw.get("_scrolled")]
        self.assertTrue(scrolled, "the page did not scroll")

    def test_focused_combobox_steps_its_value(self):
        self.write({})
        fake, app = _app(self)
        combo = app.fields["TTS_BACKEND"].combo
        fake.focus = combo
        combo.bindings["<MouseWheel>"](self._combo_event(fake, combo))
        self.assertEqual(app.fields["TTS_BACKEND"].get_raw(), "kokoro")

    def test_page_wheel_scrolls_the_selected_tab(self):
        self.write({})
        fake, app = _app(self)
        seq, handler = fake.bound_all
        self.assertEqual(seq, "<MouseWheel>")
        pages = [w for w in fake.widgets if type(w).__name__ == "Canvas"
                 and "height" not in w.kw]
        self.assertEqual(len(pages), len(sw.TAB_ORDER))
        for idx in (0, 2, len(sw.TAB_ORDER) - 1):
            for p in pages:
                p.kw.pop("_scrolled", None)
            app.notebook.select(idx)
            handler(types.SimpleNamespace(widget=None, delta=-120))
            scrolled = [i for i, p in enumerate(pages) if p.kw.get("_scrolled")]
            self.assertEqual(scrolled, [idx])

    def test_touchpad_deltas_accumulate(self):
        steps, acc = 0, 0.0
        total = 0
        for _ in range(8):
            steps, acc = sw.wheel_steps(acc, -30)
            total += steps
        self.assertEqual(total, 2)            # 8 × 30 = 240 = two notches
        self.assertEqual(sw.wheel_steps(0.0, 120), (-1, 0.0))
        self.assertEqual(sw.wheel_steps(0.0, -120), (1, 0.0))


# ════════════════════════════════════════════════════════════════════════
#  P1-5  the mic / speaker picker saves a NAME
# ════════════════════════════════════════════════════════════════════════
class DevicePickerTests(_TmpDir):
    def test_filters_to_real_input_devices(self):
        rows = sw.list_audio_devices("input", devices=_DEVICES,
                                     hostapis=_HOSTAPIS)
        labels = " | ".join(r["label"] for r in rows)
        for junk in ("Sound Mapper", "Primary Sound", "Microphone ()",
                     "Line (", "Analog Connector", "Steam Streaming"):
            self.assertNotIn(junk, labels)
        names = [r["name"] for r in rows]
        self.assertEqual(names, ["Microphone (Blue Snowball )",
                                 "Microphone Array (Xbox NUI Sensor)"])

    def test_one_row_per_device_with_its_apis(self):
        rows = sw.list_audio_devices("input", devices=_DEVICES,
                                     hostapis=_HOSTAPIS)
        snow = rows[0]
        self.assertEqual(snow["apis"], ["MME", "DirectSound", "WASAPI",
                                        "WDM-KS"])
        self.assertIn("MME · DirectSound · WASAPI · WDM-KS", snow["label"])
        self.assertIn("WDM-KS only", rows[1]["label"])

    def test_mme_truncated_output_name_is_merged(self):
        rows = sw.list_audio_devices("output", devices=_DEVICES,
                                     hostapis=_HOSTAPIS)
        u28 = [r for r in rows if r["name"].startswith("U28E510")]
        self.assertEqual(len(u28), 1)
        self.assertIn("High Definition Audio", u28[0]["label"])
        # The saved (MME, truncated) name matches every API's row.
        self.assertEqual(_pick_device([u28[0]["name"]], _DEVICES, False)[0], 4)
        self.assertNotIn("Output ()", " ".join(r["label"] for r in rows))

    def test_saved_name_survives_renumbering(self):
        rows = sw.list_audio_devices("input", devices=_DEVICES,
                                     hostapis=_HOSTAPIS)
        choices, initial = sw.audio_device_choices(
            "input", None, [], devices=_DEVICES, hostapis=_HOSTAPIS)
        snow = [c for c in choices if c["kind"] == "name"
                and "Blue Snowball" in c["label"]][0]
        new_list = sw.device_choice_list(snow, None, [])
        self.assertEqual(sw.device_choice_index(snow), None)
        # A USB device appears in front of everything: every index shifts.
        shifted = [_dev("Microphone (New USB Headset)", 0, 1, 0)] + _DEVICES
        idx, name = _pick_device(new_list, shifted, True)
        self.assertEqual(name, "Microphone (Blue Snowball )")
        self.assertEqual(idx, 2)
        self.assertEqual(rows[0]["indices"][0], 1)

    def test_choices_and_initial_selection(self):
        ch, init = sw.audio_device_choices("input", None, [],
                                           devices=_DEVICES,
                                           hostapis=_HOSTAPIS)
        self.assertEqual(init["kind"], "auto")
        self.assertEqual([c["kind"] for c in ch[:2]], ["auto", "off"])
        _ch, init = sw.audio_device_choices("input", -1, [],
                                            devices=_DEVICES,
                                            hostapis=_HOSTAPIS)
        self.assertEqual(init["kind"], "off")
        _ch, init = sw.audio_device_choices("input", None, ["snowball"],
                                            devices=_DEVICES,
                                            hostapis=_HOSTAPIS)
        self.assertEqual((init["kind"], init["value"]),
                         ("name", "Microphone (Blue Snowball )"))
        ch, init = sw.audio_device_choices("input", None, ["Yeti"],
                                           devices=_DEVICES,
                                           hostapis=_HOSTAPIS)
        self.assertIn("not connected", init["label"])
        ch, init = sw.audio_device_choices("input", 12, [],
                                           devices=_DEVICES,
                                           hostapis=_HOSTAPIS)
        self.assertEqual((init["kind"], init["value"]), ("index", 12))
        self.assertIn("pinned by number", init["label"])
        out, init = sw.audio_device_choices("output", None, [],
                                            devices=_DEVICES,
                                            hostapis=_HOSTAPIS)
        self.assertNotIn("off", [c["kind"] for c in out])
        self.assertEqual(len({c["label"] for c in ch}), len(ch))

    def test_device_choice_list_replaces_only_the_owned_head(self):
        pick = {"kind": "name", "value": "Mic B"}
        self.assertEqual(sw.device_choice_list(pick, "Mic A",
                                               ["Mic A", "Typed C"]),
                         ["Mic B", "Typed C"])
        self.assertEqual(sw.device_choice_list({"kind": "auto"}, "Mic A",
                                               ["Mic A", "Typed C"]),
                         ["Typed C"])
        self.assertEqual(sw.device_choice_list({"kind": "off", "value": -1},
                                               "Mic A", ["Mic A"]), ["Mic A"])

    def test_gui_pick_saves_the_name_not_an_index(self):
        self.write({"MICROPHONE_INDEX": 12})
        fake, app = _app(self)
        st = app.device_pickers["MICROPHONE_INDEX"]
        self.assertEqual(st["choice"]["kind"], "index")
        snow = [lbl for lbl, c in st["by_label"].items()
                if c["kind"] == "name" and "Blue Snowball" in lbl][0]
        st["var"].set(snow)
        self.assertTrue(app.save())
        doc = self.read()
        self.assertIsNone(doc["MICROPHONE_INDEX"])
        self.assertEqual(doc["PREFERRED_INPUT_DEVICES"],
                         ["Microphone (Blue Snowball )"])

    def test_gui_speaker_picker_exists_and_saves_by_name(self):
        self.write({})
        fake, app = _app(self)
        st = app.device_pickers["SPEAKER_INDEX"]
        spk = [lbl for lbl, c in st["by_label"].items()
               if c["kind"] == "name" and "Realtek" in lbl][0]
        st["var"].set(spk)
        self.assertTrue(app.save())
        self.assertEqual(self.read(), {"PREFERRED_OUTPUT_DEVICES":
                                       ["Speakers (Realtek USB2.0 Audio)"]})

    def test_untouched_picker_writes_nothing(self):
        self.write({"PREFERRED_INPUT_DEVICES": ["snowball"]})
        fake, app = _app(self)
        self.assertEqual(app.collect(), ({}, {}))


# ════════════════════════════════════════════════════════════════════════
#  P1-6 / P1-7  the important invisible keys get rows
# ════════════════════════════════════════════════════════════════════════
_NEW_ROWS = [
    "KINECT_AS_CAMERA", "KINECT_PRESENCE_ENABLED", "KINECT_PRESENCE_STANDBY",
    "KINECT_PRESENCE_WAKE", "KINECT_GREET_ON_ENTRY", "KINECT_POSTURE_NUDGE",
    "KINECT_GAZE_ENABLED", "KINECT_GESTURES_ENABLED",
    "KINECT_POINT_CONTROL_ENABLED", "KINECT_AIR_MOUSE_ENABLED",
    "KINECT_TWO_HAND_ENABLED", "KINECT_GUARD_ENABLED",
    "KINECT_SKELETON_OVERLAY_ENABLED",
    "AUDIO_AUTOSWITCH_ENABLED", "AUDIO_AUTOSWITCH_HEADSET",
    "AUDIO_AUTOSWITCH_FALLBACK", "AUDIO_AUTOSWITCH_MIC",
    "AUDIO_AUTOSWITCH_MIC_FALLBACK",
    "PREFERRED_OUTPUT_DEVICES", "SPEAKER_INDEX",
    "FACE_ID_ENABLED", "GREET_NEW_PEOPLE_ENABLED",
    "DIALOGUE_ENABLED", "DIALOGUE_MAX_S", "DIALOGUE_STOP_LISTEN",
    "DIALOGUE_BEAT_S", "DIALOGUE_LOST_HOLD_S",
    "FOLLOWUP_WINDOW_S", "DEVICE_SPEECH_FILTER_ENABLED", "GAME_MODE_ENABLED",
    "TV_DETECT_ENABLED", "HUD_MONITOR", "LOCAL_VISION_MODEL",
]


class NewRowsTests(_TmpDir):
    @staticmethod
    def _shipped_config():
        """core/config.py's module values (types for env-derived ones)."""
        import core.config as cfg
        return cfg

    def test_rows_exist_and_match_core_config(self):
        cfg = self._shipped_config()
        pytype = {"bool": bool, "int": int, "float": float, "str": str,
                  "combo": str, "enum": str, "text": list}
        for key in _NEW_ROWS:
            self.assertIn(key, sw.SCHEMA, key)
            spec = sw.SCHEMA[key]
            self.assertIn(key, sw.persisted_keys(), key)
            self.assertTrue(hasattr(cfg, key), key)
            if spec["type"] == "device":
                self.assertIsNone(spec["default"], key)
                continue
            want = pytype[spec["type"]]
            self.assertIsInstance(spec["default"], want, key)
            live = getattr(cfg, key)
            if want is int:
                self.assertNotIsInstance(live, bool, key)
            self.assertTrue(isinstance(live, want)
                            or (want is list and isinstance(live, tuple)),
                            f"{key}: schema type {spec['type']} vs config "
                            f"{type(live).__name__}")

    def test_privacy_rows_are_on_the_privacy_tab(self):
        for key in ("FACE_ID_ENABLED", "GREET_NEW_PEOPLE_ENABLED"):
            self.assertEqual(sw.SCHEMA[key]["tab"], "privacy", key)

    def test_cameras_are_shown_read_only(self):
        rows = [k for k, s in sw.SCHEMA.items()
                if s.get("type") == "view" and s.get("source_key") == "CAMERAS"]
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0].startswith("_"))
        self.assertNotIn("CAMERAS", sw.persisted_keys())
        self.assertNotIn(rows[0], sw.persisted_keys())
        lines = sw.cameras_summary([{"index": 2, "label": "Left", "name": "c"}])
        self.assertIn("Left", lines[0])
        self.assertIn("index 2", lines[0])

    def test_gui_renders_the_camera_list_and_never_writes_it(self):
        self.write({"CAMERAS": [{"index": 5, "label": "Fake cam",
                                 "name": "fake"}]})
        fake, app = _app(self)
        texts = " ".join(str(w.kw.get("text", "")) for w in fake.widgets)
        self.assertIn("Fake cam", texts)
        app.fields["TTS_VOICE"].set_raw("x")
        app.save()
        self.assertEqual(self.read()["CAMERAS"],
                         [{"index": 5, "label": "Fake cam", "name": "fake"}])

    def test_the_web_panel_serves_the_new_rows(self):
        from tools import web_interface as wi
        payload = wi.build_settings_schema(self.path)
        names = {row["name"] for row in payload["settings"]}
        for key in _NEW_ROWS:
            self.assertIn(key, names, key)
        self.assertFalse(any(n.startswith("_") for n in names))


# ════════════════════════════════════════════════════════════════════════
#  P1-8  the VRAM panel respects WHISPER_DEVICE
# ════════════════════════════════════════════════════════════════════════
class VramWhisperTests(unittest.TestCase):
    def test_whisper_device_is_a_watched_budget_input(self):
        self.assertIn("WHISPER_DEVICE", sw.VRAM_WATCH_KEYS)
        got = sw.resolve_vram_values({}, {"WHISPER_DEVICE": "cuda:1"})
        self.assertEqual(got["WHISPER_DEVICE"], "cuda:1")

    def test_whisper_on_the_second_card_costs_the_brain_card_nothing(self):
        base = {"LOCAL_LLM_MODEL": "gemma4:26b-a4b-it-qat",
                "LOCAL_VISION_MODEL": "gemma4:26b-a4b-it-qat",
                "MODEL_ROUTING::vision": "local", "RAG_ENABLED": False,
                "SCREEN_VISION_ENABLED": True}
        on_1650 = sw.budget_from_live_values(dict(base,
                                                  WHISPER_DEVICE="cuda:1"),
                                             total_mb=24576)
        on_3090 = sw.budget_from_live_values(dict(base, WHISPER_DEVICE="auto"),
                                             total_mb=24576)
        self.assertEqual(on_3090["total_mb"] - on_1650["total_mb"],
                         int(1.5 * 1024))
        self.assertIn("Whisper (on cuda:1, not this card)",
                      sw.budget_parts_text(on_1650))

    def test_gui_budget_follows_the_whisper_widget(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "user_settings.json")
            with open(p, "w", encoding="utf-8") as f:
                json.dump({"WHISPER_DEVICE": "auto", "RAG_ENABLED": False}, f)
            t = types.SimpleNamespace(path=p,
                                      cmd=os.path.join(d, "cmds.json"))
            fake, app = _app(t)
            self.assertIsNotNone(app.vram)
            app.update_budget()
            before = app.vram_widgets["num"].kw["text"]
            app.fields["WHISPER_DEVICE"].set_raw("cuda:1")
            after = app.vram_widgets["num"].kw["text"]
            self.assertNotEqual(before, after)
            self.assertIn("Whisper (on cuda:1",
                          app.vram_widgets["parts"].kw["text"])


# ════════════════════════════════════════════════════════════════════════
#  B16  the VRAM bar shows HOW FAR over budget, not just "full"
# ════════════════════════════════════════════════════════════════════════
def _recording_bar_fake():
    """A FakeTk whose Canvas keeps its items, so the VRAM bar can be read
    back: create_* return ids, coords()/itemconfigure() update them."""
    fake = FakeTk()
    base = fake.tk.Canvas

    class BarCanvas(base):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.items = {}

        def _new(self, kind, coords, kw):
            item = 100 + len(self.items)
            self.items[item] = {"kind": kind, "coords": list(coords),
                                "kw": dict(kw)}
            return item

        def create_rectangle(self, *c, **kw):
            return self._new("rectangle", c, kw)

        def create_line(self, *c, **kw):
            return self._new("line", c, kw)

        def create_text(self, *c, **kw):
            return self._new("text", c, kw)

        def coords(self, item, *c):
            self.items[item]["coords"] = list(c)

        def itemconfigure(self, item, **kw):
            self.items[item]["kw"].update(kw)

        itemconfig = itemconfigure

    fake.tk.Canvas = BarCanvas
    return fake


class VramBarOverageTests(_TmpDir):
    """GUI_REVIEW B16: over budget the bar's fill clamped at full width while
    the number beside it could read 137%, so 101% and 137% looked the same.
    Over budget the bar now spans the predicted peak, with a limit mark where
    the budget runs out, the overage hatched past it, and a "+N GB over"
    label on the bar."""

    _CW = 700   # FakeTk widgets report winfo_width() == 700

    def _bar(self, total_vram_mb):
        self.write({"WHISPER_DEVICE": "auto", "RAG_ENABLED": False})
        fake, app = _app(self, fake=_recording_bar_fake(),
                         total_vram_mb=total_vram_mb)
        self.assertIsNotNone(app.vram)
        app.update_budget()
        return app

    @staticmethod
    def _visible(canvas):
        return [it for it in canvas.items.values()
                if it["kw"].get("state") != "hidden"]

    def _budget(self, app):
        wv = app.widget_values()
        widget = {k: wv[k] for k in sw.VRAM_WATCH_KEYS if k in wv}
        return sw.budget_from_live_values(
            sw.resolve_vram_values(widget, app.values),
            total_mb=app.vram_total_mb)

    def test_over_budget_bar_marks_the_limit_and_labels_the_overage(self):
        app = self._bar(total_vram_mb=8192)
        b = self._budget(app)
        self.assertTrue(b["over"])
        canvas = app.vram_widgets["canvas"]
        limit_x = int(self._CW * b["budget_mb"] / b["total_mb"])
        over_gb = (b["total_mb"] - b["budget_mb"]) / 1024.0
        texts = [it["kw"].get("text", "") for it in self._visible(canvas)
                 if it["kind"] == "text"]
        self.assertTrue(any(f"+{over_gb:.1f} GB over" in t for t in texts),
                        texts)
        marks = [it for it in self._visible(canvas) if it["kind"] == "line"
                 and it["coords"][0] == it["coords"][2] == limit_x]
        self.assertEqual(len(marks), 1, "no limit mark where the budget ends")
        # The solid fill stops at the limit; the overage past it is hatched
        # and runs to the end of the bar.
        fill = canvas.items[app.vram_widgets["bar_fill"]]
        self.assertEqual(fill["coords"][2], limit_x)
        hatched = [it for it in self._visible(canvas)
                   if it["kind"] == "rectangle" and it["kw"].get("stipple")]
        self.assertEqual([h["coords"][0] for h in hatched], [limit_x])
        self.assertEqual([h["coords"][2] for h in hatched], [self._CW])

    def test_within_budget_bar_has_no_overage_marks(self):
        app = self._bar(total_vram_mb=24576)
        b = self._budget(app)
        self.assertFalse(b["over"])
        canvas = app.vram_widgets["canvas"]
        visible = self._visible(canvas)
        self.assertFalse([it for it in visible if it["kind"] in
                          ("line", "text")])
        fill = canvas.items[app.vram_widgets["bar_fill"]]
        self.assertEqual(fill["coords"][2],
                         int(self._CW * b["total_mb"] / b["budget_mb"]))

    def test_marks_follow_the_budget_back_under(self):
        app = self._bar(total_vram_mb=8192)
        app.vram_total_mb = 24576
        app.update_budget()
        canvas = app.vram_widgets["canvas"]
        self.assertFalse([it for it in self._visible(canvas)
                          if it["kind"] in ("line", "text")])

    def test_layout_helper(self):
        lay = sw.vram_bar_layout({"total_mb": 137, "budget_mb": 100}, 700)
        self.assertEqual(lay["limit_px"], int(700 * 100 / 137))
        self.assertEqual(lay["fill_px"], lay["limit_px"])
        self.assertIn("over", lay["over_label"])
        lay = sw.vram_bar_layout({"total_mb": 50, "budget_mb": 100}, 700)
        self.assertEqual((lay["fill_px"], lay["limit_px"], lay["over_label"]),
                         (350, None, ""))
        self.assertEqual(sw.vram_bar_layout({}, 700)["fill_px"], 0)
        self.assertEqual(sw.vram_bar_layout(None, 0)["limit_px"], None)

    def test_layout_reports_which_case_it_drew(self):
        self.assertTrue(sw.vram_bar_layout(
            {"total_mb": 137, "budget_mb": 100}, 700)["over"])
        self.assertFalse(sw.vram_bar_layout(
            {"total_mb": 100, "budget_mb": 100}, 700)["over"])
        self.assertFalse(sw.vram_bar_layout({}, 700)["over"])
        self.assertFalse(sw.vram_bar_layout({"total_mb": "x"}, 700)["over"])

    def test_zero_budget_card_is_all_overage_not_an_empty_bar(self):
        # 2026-10-02: a card at or under the headroom has a usable budget of
        # 0. predict_budget() calls any load on it over, but the layout drew
        # an empty, unmarked bar — the same picture as nothing loaded at all.
        lay = sw.vram_bar_layout({"total_mb": 2048, "budget_mb": 0}, 700)
        self.assertTrue(lay["over"])
        self.assertEqual((lay["fill_px"], lay["limit_px"]), (0, 0))
        self.assertEqual(lay["over_label"], "+2.0 GB over")
        # Nothing loaded on that card is still an empty, unmarked bar.
        lay = sw.vram_bar_layout({"total_mb": 0, "budget_mb": 0}, 700)
        self.assertEqual((lay["fill_px"], lay["limit_px"], lay["over"]),
                         (0, None, False))

    def test_zero_budget_card_paints_the_hatched_overage(self):
        # 1 GB card: below the ~1.5 GB headroom, so the usable budget is 0.
        app = self._bar(total_vram_mb=1024)
        b = self._budget(app)
        self.assertEqual(b["budget_mb"], 0)
        self.assertTrue(b["over"])
        canvas = app.vram_widgets["canvas"]
        visible = self._visible(canvas)
        hatched = [it for it in visible
                   if it["kind"] == "rectangle" and it["kw"].get("stipple")]
        self.assertEqual([(h["coords"][0], h["coords"][2]) for h in hatched],
                         [(0, self._CW)])
        texts = [it["kw"].get("text", "") for it in visible
                 if it["kind"] == "text"]
        over_gb = b["total_mb"] / 1024.0
        self.assertTrue(any(f"+{over_gb:.1f} GB over" in t for t in texts),
                        texts)


# ════════════════════════════════════════════════════════════════════════
#  P1-9  rows whose setting isn't actually in effect say so
# ════════════════════════════════════════════════════════════════════════
class EffectiveStateTests(_TmpDir):
    @staticmethod
    def _spec_without(*missing):
        return lambda name: None if name in missing else object()

    def test_realtime_without_pyaudio(self):
        notes = sw.effective_warnings({"VOICE_MODE": "realtime"},
                                      find_spec=self._spec_without("pyaudio"))
        self.assertIn("pyaudio", notes["VOICE_MODE"])
        self.assertIn("turn_based", notes["VOICE_MODE"])
        self.assertNotIn("VOICE_MODE", sw.effective_warnings(
            {"VOICE_MODE": "realtime"}, find_spec=self._spec_without()))

    def test_xtts_without_its_package(self):
        notes = sw.effective_warnings({"TTS_BACKEND": "xtts"},
                                      find_spec=self._spec_without("TTS"))
        self.assertIn("TTS_BACKEND", notes)

    def test_barge_in_needs_the_detector(self):
        notes = sw.effective_warnings({"BARGE_IN_ENABLED": True},
                                      find_spec=self._spec_without())
        self.assertIn("start listening for the wake word",
                      notes["BARGE_IN_ENABLED"])
        self.assertNotIn("BARGE_IN_ENABLED", sw.effective_warnings(
            {"BARGE_IN_ENABLED": False}, find_spec=self._spec_without()))

    def test_gui_shows_the_note_and_updates_it_live(self):
        self.write({"VOICE_MODE": "realtime"})
        fake, app = _app(self, find_spec=self._spec_without("pyaudio"))
        note = app.fields["VOICE_MODE"].note_label
        self.assertTrue(note.visible)
        self.assertIn("pyaudio", note.kw["text"])
        app.fields["VOICE_MODE"].set_raw("turn_based")
        self.assertFalse(note.visible)


# ════════════════════════════════════════════════════════════════════════
#  P1-10  the web token is masked
# ════════════════════════════════════════════════════════════════════════
class TokenMaskTests(_TmpDir):
    def test_run_gui_masks_the_token_entry(self):
        self.write({"WEB_INTERFACE_TOKEN": "fake-token-123"})
        fake = FakeTk()
        _run_gui_with_fake_tk(fake, self.path)
        entries = [w for w in fake.widgets
                   if w.kw.get("textvariable") is not None
                   and getattr(w.kw["textvariable"], "_v", None)
                   == "fake-token-123"]
        self.assertTrue(entries)
        self.assertTrue(all(w.kw.get("show") for w in entries),
                        "the web-interface token is shown in clear text")


# ════════════════════════════════════════════════════════════════════════
#  P1-11 / P1-12  tabs, sections and help that tell the truth
# ════════════════════════════════════════════════════════════════════════
class LayoutAndHelpTests(unittest.TestCase):
    def test_rows_live_on_their_tabs(self):
        where = {"KINECT_ENABLED": "cameras", "AIR_MOUSE_REQUIRE_OPEN_PALM":
                 "cameras", "CAMERA_OPEN_MIN_GAP_S": "cameras",
                 "DAILY_BUDGET_USD": "ai", "DEEP_AUDIT_BUDGET_USD": "ai",
                 "MICROPHONE_INDEX": "hearing", "WHISPER_DEVICE": "hearing",
                 "VAD_THRESHOLD": "hearing", "TTS_VOICE": "voice",
                 "MODEL_ROUTING": "ai", "STREAMING_AUTO_FULLSCREEN":
                 "integrations", "WEB_INTERFACE_TOKEN": "advanced"}
        for key, tab in where.items():
            self.assertEqual(sw.SCHEMA[key]["tab"], tab, key)

    def test_layout_lists_every_row_once_under_its_own_tab(self):
        seen = {}
        for tab, sections in sw.TAB_SECTIONS.items():
            self.assertIn(tab, sw.TAB_ORDER)
            for name, keys in sections:
                self.assertTrue(name)
                for key in keys:
                    self.assertIn(key, sw.SCHEMA, key)
                    self.assertEqual(sw.SCHEMA[key]["tab"], tab, key)
                    self.assertNotIn(key, seen, key)
                    seen[key] = name
        self.assertEqual(set(seen), set(sw.SCHEMA),
                         "rows missing from TAB_SECTIONS render under 'Other'")
        for key in sw.persisted_keys():
            self.assertNotEqual(sw.row_section(key), "Other", key)

    def test_web_panel_lists_tabs_in_the_window_order(self):
        # web_interface orders its groups by first appearance in SCHEMA.
        from tools import web_interface as wi
        with tempfile.TemporaryDirectory() as d:
            tabs = wi.build_settings_schema(
                os.path.join(d, "none.json"))["tabs"]
        self.assertEqual(tabs, sw.TAB_ORDER)

    def test_stale_help_is_gone(self):
        self.assertNotIn("3.1 GB", sw.SCHEMA["WHISPER_MODEL_CUDA"]["help"])
        self.assertIn("1.5", sw.SCHEMA["WHISPER_MODEL_CUDA"]["help"])
        self.assertNotIn("VRAM)", sw.SCHEMA["LTM_ENABLED"]["help"])
        self.assertIn("CPU", sw.SCHEMA["LTM_ENABLED"]["help"])
        self.assertNotIn("7.3 GB", sw.SCHEMA["LOCAL_VISION_FALLBACK"]["help"])
        self.assertNotIn("32B", sw.SCHEMA["LOCAL_LLM_MODEL"]["help"])
        self.assertIn("doubling",
                      sw.SCHEMA["CAMERA_REOPEN_MAX_BACKOFF_S"]["help"])
        doc = sw.__doc__ or ""
        self.assertNotIn("SEPARATE, parallel task", doc)
        self.assertNotIn("creates\n  ``data/user_settings.json`` from the "
                         "built-in defaults", doc)

    def test_offline_model_list_starts_with_the_default_brain(self):
        # The offline list's first entry was commented "default" while being a
        # different tag from the real default.
        import ast
        with open(os.path.join(_PROJECT, "core", "config.py"),
                  encoding="utf-8") as f:
            tree = ast.parse(f.read())
        shipped = None
        for node in tree.body:
            if (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and getattr(node.targets[0], "id", "") == "LOCAL_LLM_MODEL"):
                shipped = ast.literal_eval(node.value)
        self.assertEqual(sw.OLLAMA_MODEL_FALLBACK[0], shipped)
        self.assertEqual(sw.SCHEMA["LOCAL_LLM_MODEL"]["default"], shipped)


# ════════════════════════════════════════════════════════════════════════
#  P1-13  one window, brought to the front, never pinned on top
# ════════════════════════════════════════════════════════════════════════
class SingleInstanceTests(_TmpDir):
    def test_run_gui_is_not_always_on_top(self):
        fake = FakeTk()
        _run_gui_with_fake_tk(fake, self.path)
        root = fake.roots[0]
        root.run_afters()
        tops = [a for a in root.attrs if a and a[0] == "-topmost"]
        self.assertTrue(not tops or tops[-1] == ("-topmost", False),
                        "the Settings window is left always-on-top, covering "
                        "the editor 'Open user_settings.json' starts")

    @unittest.skipUnless(sys.platform == "win32", "named mutex is Windows-only")
    def test_second_instance_is_refused(self):
        tag = f"unittest-{os.getpid()}-{time.time_ns()}"
        first = sw.acquire_single_instance(tag)
        try:
            self.assertTrue(first)
            self.assertIsNone(sw.acquire_single_instance(tag))
        finally:
            sw.release_single_instance(first)
        again = sw.acquire_single_instance(tag)
        self.assertTrue(again)
        sw.release_single_instance(again)

    def test_focus_request_round_trip(self):
        tag = "unittest-focus"
        self.assertFalse(sw.consume_focus_request(tag, self.dir))
        self.assertTrue(sw.request_focus(tag, self.dir))
        self.assertTrue(sw.consume_focus_request(tag, self.dir))
        self.assertFalse(sw.consume_focus_request(tag, self.dir))

    def test_open_window_answers_a_focus_request(self):
        fake, app = _app(self, focus_tag="unittest-open", focus_dir=self.dir)
        lifted = app.root.lifted
        sw.request_focus("unittest-open", self.dir)
        app._tick()
        self.assertGreater(app.root.lifted, lifted)

    def test_main_hands_off_to_the_open_window(self):
        with mock.patch.object(sw, "acquire_single_instance",
                               return_value=None), \
                mock.patch.object(sw, "request_focus") as req, \
                mock.patch.object(sw, "run_gui") as run:
            self.assertEqual(sw.main([]), 0)
        req.assert_called_once()
        run.assert_not_called()

    def test_open_json_uses_the_opener(self):
        opened = []
        fake, app = _app(self, file_opener=opened.append)
        app.open_json()
        self.assertEqual(opened, [self.path])
        self.assertTrue(os.path.exists(self.path))
        self.assertEqual(self.read(), {})


# ════════════════════════════════════════════════════════════════════════
#  P1-14  OBS: every env var is optional
# ════════════════════════════════════════════════════════════════════════
class ObsStatusTests(unittest.TestCase):
    def test_obs_with_no_env_is_ready_on_defaults(self):
        spec = sw.SCHEMA["_status_obs"]
        env = {k: "" for k in ("OBS_HOST", "OBS_PORT", "OBS_PASSWORD")}
        with mock.patch.dict(os.environ, env):
            present, detail = sw.integration_status(
                spec, find_spec=lambda m: object())
        self.assertTrue(present, "OBS reads 'not set' with the defaults, "
                                 "which work")
        self.assertIn("127.0.0.1:4455", detail)

    def test_obs_names_a_missing_client_package(self):
        present, detail = sw.integration_status(sw.SCHEMA["_status_obs"],
                                                find_spec=lambda m: None)
        self.assertFalse(present)
        self.assertIn("obs-websocket-py", detail)

    def test_obs_reports_which_overrides_are_set_never_values(self):
        with mock.patch.dict(os.environ, {"OBS_PASSWORD": "fake-secret",
                                          "OBS_HOST": "", "OBS_PORT": ""}):
            present, detail = sw.integration_status(
                sw.SCHEMA["_status_obs"], find_spec=lambda m: object())
        self.assertTrue(present)
        self.assertIn("OBS_PASSWORD set", detail)
        self.assertNotIn("fake-secret", detail)


# ════════════════════════════════════════════════════════════════════════
#  P2  keyboard, unsaved changes, async model list
# ════════════════════════════════════════════════════════════════════════
class ConvenienceTests(_TmpDir):
    def test_keys_are_bound(self):
        fake, app = _app(self)
        for seq in ("<Control-s>", "<Escape>"):
            self.assertIn(seq, app.root.bindings)
        self.assertIn("WM_DELETE_WINDOW", app.root.protocols)

    def test_close_with_changes_asks_first(self):
        self.write({})
        fake, app = _app(self)
        app.fields["TTS_VOICE"].set_raw("x")
        fake.ask_answer = None                 # Cancel
        app.close()
        self.assertFalse(app.root.destroyed)
        fake.ask_answer = False                # Don't save
        app.close()
        self.assertTrue(app.root.destroyed)
        self.assertEqual(self.read(), {})

    def test_close_with_yes_saves(self):
        self.write({})
        fake, app = _app(self)
        app.fields["TTS_VOICE"].set_raw("x")
        fake.ask_answer = True
        app.close()
        self.assertTrue(app.root.destroyed)
        self.assertEqual(self.read(), {"TTS_VOICE": "x"})

    def test_close_without_changes_does_not_ask(self):
        fake, app = _app(self)
        app.close()
        self.assertEqual(fake.asked, [])
        self.assertTrue(app.root.destroyed)

    def test_model_list_arrives_in_the_background(self):
        self.write({})
        fake, app = _app(self)
        app._apply_async()
        chat = app.fields["LOCAL_LLM_MODEL"].combo.kw["values"]
        vision = app.fields["LOCAL_VISION_MODEL"].combo.kw["values"]
        self.assertIn("gemma4:12b", chat)
        self.assertNotIn("qwen2.5vl:7b", chat)
        self.assertIn("qwen2.5vl:7b", vision)
        self.assertIn("off", vision)

    def test_tabs_are_built_in_order(self):
        fake, app = _app(self)
        self.assertEqual(app.notebook.tabs,
                         [sw.TAB_LABELS[t] for t in sw.TAB_ORDER])


if __name__ == "__main__":
    unittest.main()
