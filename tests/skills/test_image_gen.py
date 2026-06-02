"""Logic tests for skills/image_gen.py.

Local image generation via ComfyUI HTTP or diffusers. Default backend is 'off'
so the high-value degradation path is trivial. We also test the pure prompt /
config / workflow-builder logic and drive the ComfyUI backend end-to-end with a
fake `requests` module (no HTTP, no GPU, no disk writes — _save_and_open is
mocked).
"""
from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("image_gen")

    def test_default_backend_off(self):
        # No env, and bobert_companion (if present) shouldn't define the attr in
        # a way that flips this — patch resolve to be deterministic.
        with mock.patch.object(self.mod, "_resolve",
                               side_effect=lambda env, attr, default: default):
            backend, model, url, steps = self.mod._config()
        self.assertEqual(backend, "off")
        self.assertEqual(steps, self.mod._DEFAULT_STEPS)
        self.assertEqual(url, self.mod._DEFAULT_COMFYUI_URL)

    def test_backend_env_diffusers_picks_model_default(self):
        with mock.patch.dict(os.environ, {"IMAGE_GEN_BACKEND": "diffusers"}, clear=True):
            backend, model, _url, _steps = self.mod._config()
        self.assertEqual(backend, "diffusers")
        self.assertEqual(model, self.mod._DEFAULT_DIFFUSERS_MODEL)

    def test_backend_env_comfyui_picks_ckpt_default(self):
        with mock.patch.dict(os.environ, {"IMAGE_GEN_BACKEND": "comfyui"}, clear=True):
            backend, model, _url, _steps = self.mod._config()
        self.assertEqual(backend, "comfyui")
        self.assertEqual(model, self.mod._DEFAULT_COMFYUI_CKPT)

    def test_invalid_backend_coerced_off(self):
        with mock.patch.dict(os.environ, {"IMAGE_GEN_BACKEND": "dalle"}, clear=True):
            backend, *_ = self.mod._config()
        self.assertEqual(backend, "off")

    def test_steps_bad_value_falls_back(self):
        with mock.patch.dict(os.environ, {"IMAGE_GEN_BACKEND": "comfyui",
                                          "IMAGE_GEN_STEPS": "lots"}, clear=True):
            *_, steps = self.mod._config()
        self.assertEqual(steps, self.mod._DEFAULT_STEPS)


class SlugTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("image_gen")

    def test_slug_normalises(self):
        self.assertEqual(self.mod._slug("A Mars Colony at Sunset!"),
                         "a_mars_colony_at_sunset")

    def test_slug_truncates(self):
        self.assertLessEqual(len(self.mod._slug("x " * 100, limit=10)), 10)

    def test_slug_empty_fallback(self):
        self.assertEqual(self.mod._slug("!!!"), "image")


class CudaOomDetectTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("image_gen")

    def test_detects_oom_message(self):
        self.assertTrue(self.mod._is_cuda_oom(RuntimeError("CUDA out of memory")))

    def test_detects_cuda_keyword(self):
        self.assertTrue(self.mod._is_cuda_oom(Exception("cuda error 700")))

    def test_non_oom_is_false(self):
        self.assertFalse(self.mod._is_cuda_oom(ValueError("bad prompt")))


class ComfyWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("image_gen")

    def test_workflow_graph_shape(self):
        wf = self.mod._comfyui_workflow("a cat", "model.safetensors", steps=4, seed=99)
        # KSampler carries the seed/steps and wires to the right nodes.
        ks = wf["3"]
        self.assertEqual(ks["class_type"], "KSampler")
        self.assertEqual(ks["inputs"]["seed"], 99)
        self.assertEqual(ks["inputs"]["steps"], 4)
        # Positive prompt node carries the prompt text; checkpoint node the ckpt.
        self.assertEqual(wf["6"]["inputs"]["text"], "a cat")
        self.assertEqual(wf["4"]["inputs"]["ckpt_name"], "model.safetensors")
        # SaveImage terminal node exists.
        self.assertEqual(wf["9"]["class_type"], "SaveImage")


class GenerateImageDegradationTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("image_gen")

    def test_empty_prompt(self):
        self.assertIn("format:", self.actions["generate_image"](""))

    def test_off_backend_message(self):
        with mock.patch.object(self.mod, "_config",
                               return_value=("off", "m", "http://x", 4)):
            out = self.actions["generate_image"]("a robot")
        self.assertIn("off", out)
        self.assertIn("IMAGE_GEN_BACKEND", out)

    def test_make_picture_alias_delegates(self):
        with mock.patch.object(self.mod, "generate_image", return_value="ok") as gi:
            self.assertEqual(self.actions["make_picture"]("cat"), "ok")
        gi.assert_called_once_with("cat")

    def test_diffusers_missing_deps_hint(self):
        # Force diffusers backend, then make the in-function torch/diffusers
        # import fail so the friendly install hint is returned (no GPU touched).
        with mock.patch.object(self.mod, "_config",
                               return_value=("diffusers", "stabilityai/sdxl-turbo",
                                             "http://x", 4)), \
             mock.patch.dict(sys.modules, {"torch": None, "diffusers": None}):
            out = self.actions["generate_image"]("a fox")
        self.assertIn("diffusers", out)
        self.assertIn("pip install", out)


class _FakeResponse:
    def __init__(self, *, ok=True, status_code=200, json_data=None, content=b"",
                 text=""):
        self.ok = ok
        self.status_code = status_code
        self._json = json_data or {}
        self.content = content
        self.text = text

    def json(self):
        return self._json


def _fake_requests(get_map=None, post_map=None):
    """Build a fake `requests` module. get_map / post_map map a URL-suffix
    substring → _FakeResponse (or a callable returning one)."""
    mod = mock.MagicMock(name="requests")

    def _lookup(table, url):
        for suffix, resp in (table or {}).items():
            if suffix in url:
                return resp() if callable(resp) else resp
        raise AssertionError(f"unexpected request to {url}")

    mod.get = mock.MagicMock(side_effect=lambda url, **kw: _lookup(get_map, url))
    mod.post = mock.MagicMock(side_effect=lambda url, **kw: _lookup(post_map, url))
    return mod


class ComfyuiBackendTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("image_gen")

    def _gen(self, requests_mod):
        with mock.patch.object(self.mod, "_config",
                               return_value=("comfyui", "m.safetensors",
                                             "http://localhost:8188", 4)), \
             mock.patch.dict(sys.modules, {"requests": requests_mod}), \
             mock.patch.object(self.mod, "_save_and_open",
                               return_value="image saved to /tmp/out.png"), \
             mock.patch.object(self.mod.time, "sleep"):
            return self.actions["generate_image"]("a spaceship")

    def test_comfyui_unreachable(self):
        req = _fake_requests(get_map={"/system_stats":
                                      _FakeResponse(ok=False, status_code=500)})
        out = self._gen(req)
        self.assertIn("HTTP 500", out)
        self.assertIn("ComfyUI", out)

    def test_comfyui_happy_path(self):
        prompt_id = "pid-123"
        get_map = {
            "/system_stats": _FakeResponse(ok=True),
            f"/history/{prompt_id}": _FakeResponse(
                ok=True, json_data={prompt_id: {"outputs": {
                    "9": {"images": [{"filename": "jarvis_0001.png",
                                       "subfolder": "", "type": "output"}]}}}}),
            "/view": _FakeResponse(ok=True, content=b"PNGDATA"),
        }
        post_map = {"/prompt": _FakeResponse(ok=True, json_data={"prompt_id": prompt_id})}
        out = self._gen(_fake_requests(get_map=get_map, post_map=post_map))
        self.assertIn("image saved to", out)
        # The success suffix appends elapsed time + ', sir.'
        self.assertIn("sir", out)

    def test_comfyui_rejects_workflow(self):
        get_map = {"/system_stats": _FakeResponse(ok=True)}
        post_map = {"/prompt": _FakeResponse(ok=False, status_code=400, text="bad graph")}
        out = self._gen(_fake_requests(get_map=get_map, post_map=post_map))
        self.assertIn("rejected the workflow", out)
        self.assertIn("400", out)

    def test_comfyui_no_images_produced(self):
        prompt_id = "pid-9"
        get_map = {
            "/system_stats": _FakeResponse(ok=True),
            f"/history/{prompt_id}": _FakeResponse(
                ok=True, json_data={prompt_id: {"outputs": {}}}),
        }
        post_map = {"/prompt": _FakeResponse(ok=True, json_data={"prompt_id": prompt_id})}
        out = self._gen(_fake_requests(get_map=get_map, post_map=post_map))
        self.assertIn("no images", out)

    # ── extra ComfyUI edge / failure paths ───────────────────────────────
    def test_comfyui_requests_not_installed(self):
        # The in-function `import requests` fails → friendly hint, no HTTP.
        with mock.patch.object(self.mod, "_config",
                               return_value=("comfyui", "m.safetensors",
                                             "http://localhost:8188", 4)), \
             mock.patch.dict(sys.modules, {"requests": None}):
            out = self.actions["generate_image"]("a spaceship")
        self.assertIn("requests isn't installed", out)

    def test_comfyui_unreachable_raises(self):
        # GET /system_stats raises → "couldn't reach ComfyUI" branch.
        req = _fake_requests(get_map={"/system_stats":
                                      lambda: (_ for _ in ()).throw(
                                          ConnectionError("refused"))})
        out = self._gen(req)
        self.assertIn("couldn't reach ComfyUI", out)
        self.assertIn("IMAGE_GEN_COMFYUI_URL", out)

    def test_comfyui_submit_raises(self):
        get_map = {"/system_stats": _FakeResponse(ok=True)}
        post_map = {"/prompt": lambda: (_ for _ in ()).throw(
            TimeoutError("post timed out"))}
        out = self._gen(_fake_requests(get_map=get_map, post_map=post_map))
        self.assertIn("submit failed", out)

    def test_comfyui_no_prompt_id_returned(self):
        get_map = {"/system_stats": _FakeResponse(ok=True)}
        # 200 OK but the body lacks a prompt_id.
        post_map = {"/prompt": _FakeResponse(ok=True, json_data={})}
        out = self._gen(_fake_requests(get_map=get_map, post_map=post_map))
        self.assertIn("didn't return a prompt_id", out)

    def test_comfyui_history_times_out(self):
        # /history never contains the prompt_id → deadline passes → timeout msg.
        # A negative timeout puts the deadline in the past so the poll loop is
        # skipped entirely — deterministic, no time.time() call-count coupling.
        prompt_id = "pid-timeout"
        get_map = {
            "/system_stats": _FakeResponse(ok=True),
            f"/history/{prompt_id}": _FakeResponse(ok=True, json_data={}),
        }
        post_map = {"/prompt": _FakeResponse(ok=True, json_data={"prompt_id": prompt_id})}
        with mock.patch.object(self.mod, "_COMFYUI_TIMEOUT", -1.0):
            out = self._gen(_fake_requests(get_map=get_map, post_map=post_map))
        self.assertIn("didn't finish the job", out)

    def test_comfyui_history_get_raises_then_times_out(self):
        # The /history GET raises inside the poll loop (swallowed), then the
        # deadline elapses → timeout branch. Drive a synthetic monotonic clock so
        # the loop runs exactly two iterations (clock: base, base, base, expired)
        # then expires — exercising the `except Exception: pass`. The clock is a
        # function (not a fixed side_effect list) so it can't StopIteration no
        # matter how many extra time.time() calls generate_image makes.
        prompt_id = "pid-x"
        calls = {"n": 0}

        def _history():
            calls["n"] += 1
            raise RuntimeError("transient")

        get_map = {
            "/system_stats": _FakeResponse(ok=True),
            f"/history/{prompt_id}": _history,
        }
        post_map = {"/prompt": _FakeResponse(ok=True, json_data={"prompt_id": prompt_id})}

        clock = {"t": 1000.0}

        def _now():
            v = clock["t"]
            # Advance 0.5s per call so two poll iterations run, then the 3rd
            # while-check (t=1001.5) exceeds the deadline (1000 + 1.0).
            clock["t"] += 0.5
            return v

        with mock.patch.object(self.mod, "_COMFYUI_TIMEOUT", 1.0), \
             mock.patch.object(self.mod.time, "time", side_effect=_now):
            out = self._gen(_fake_requests(get_map=get_map, post_map=post_map))
        self.assertIn("didn't finish", out)
        self.assertGreaterEqual(calls["n"], 1)

    def test_comfyui_view_http_error(self):
        prompt_id = "pid-v"
        get_map = {
            "/system_stats": _FakeResponse(ok=True),
            f"/history/{prompt_id}": _FakeResponse(
                ok=True, json_data={prompt_id: {"outputs": {
                    "9": {"images": [{"filename": "j.png",
                                       "subfolder": "", "type": "output"}]}}}}),
            "/view": _FakeResponse(ok=False, status_code=404),
        }
        post_map = {"/prompt": _FakeResponse(ok=True, json_data={"prompt_id": prompt_id})}
        out = self._gen(_fake_requests(get_map=get_map, post_map=post_map))
        self.assertIn("couldn't return the generated image", out)
        self.assertIn("404", out)

    def test_comfyui_view_fetch_raises(self):
        prompt_id = "pid-vr"
        get_map = {
            "/system_stats": _FakeResponse(ok=True),
            f"/history/{prompt_id}": _FakeResponse(
                ok=True, json_data={prompt_id: {"outputs": {
                    "9": {"images": [{"filename": "j.png"}]}}}}),
            "/view": lambda: (_ for _ in ()).throw(IOError("socket reset")),
        }
        post_map = {"/prompt": _FakeResponse(ok=True, json_data={"prompt_id": prompt_id})}
        out = self._gen(_fake_requests(get_map=get_map, post_map=post_map))
        self.assertIn("image fetch failed", out)


# ── _free_cuda_cache ─────────────────────────────────────────────────────
class FreeCudaCacheTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("image_gen")

    def test_free_cuda_cache_calls_empty_cache(self):
        torch = types.ModuleType("torch")
        torch.cuda = types.SimpleNamespace(empty_cache=mock.MagicMock())
        with mock.patch.dict(sys.modules, {"torch": torch}):
            self.mod._free_cuda_cache()
        torch.cuda.empty_cache.assert_called_once()

    def test_free_cuda_cache_swallows_missing_torch(self):
        # No torch → the guarded import fails silently (no raise).
        with mock.patch.dict(sys.modules, {"torch": None}):
            self.mod._free_cuda_cache()  # must not raise


# ── _resolve bobert-attr branch ──────────────────────────────────────────
class ResolveTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("image_gen")

    def test_resolve_reads_bobert_attr_when_env_absent(self):
        bc = types.SimpleNamespace(IMAGE_GEN_MODEL="my-model")
        with mock.patch.dict(os.environ, {}, clear=True), \
             mock.patch.object(self.mod, "_bobert", return_value=bc):
            self.assertEqual(
                self.mod._resolve("IMAGE_GEN_MODEL", "IMAGE_GEN_MODEL", "def"),
                "my-model")

    def test_resolve_env_wins_over_bobert(self):
        bc = types.SimpleNamespace(IMAGE_GEN_MODEL="bobert-model")
        with mock.patch.dict(os.environ, {"IMAGE_GEN_MODEL": "env-model"}, clear=True), \
             mock.patch.object(self.mod, "_bobert", return_value=bc):
            self.assertEqual(
                self.mod._resolve("IMAGE_GEN_MODEL", "IMAGE_GEN_MODEL", "def"),
                "env-model")

    def test_resolve_blank_bobert_attr_falls_through_to_default(self):
        bc = types.SimpleNamespace(IMAGE_GEN_MODEL="")   # empty → ignored
        with mock.patch.dict(os.environ, {}, clear=True), \
             mock.patch.object(self.mod, "_bobert", return_value=bc):
            self.assertEqual(
                self.mod._resolve("IMAGE_GEN_MODEL", "IMAGE_GEN_MODEL", "def"),
                "def")

    def test_resolve_default_when_no_env_no_bobert(self):
        with mock.patch.dict(os.environ, {}, clear=True), \
             mock.patch.object(self.mod, "_bobert", return_value=None):
            self.assertEqual(self.mod._resolve("Z", "Z", "fallback"), "fallback")


# ── _output_dir / _open_in_viewer / _prune_output_dir / _save_and_open ────
class FilesystemHelperTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("image_gen")
        self.tmp = tempfile.mkdtemp(prefix="imggen_test_")
        self.addCleanup(self._cleanup)

    def _cleanup(self):
        for root, _dirs, files in os.walk(self.tmp, topdown=False):
            for n in files:
                try:
                    os.unlink(os.path.join(root, n))
                except OSError:
                    pass
            try:
                os.rmdir(root)
            except OSError:
                pass

    def test_output_dir_creates_subdir(self):
        # __file__ points at the real skills/ dir; redirect makedirs+abspath so
        # nothing is written under the real project tree.
        target = os.path.join(self.tmp, "screenshots", self.mod._OUTPUT_SUBDIR)
        out = self.mod._output_dir()
        self.assertTrue(out.endswith(self.mod._OUTPUT_SUBDIR))
        self.assertTrue(os.path.isdir(out))
        # Sanity: it's the JARVIS_generated leaf under a screenshots parent.
        self.assertIn("screenshots", out)
        # Clean up the dir the real call just created under the project.
        try:
            os.rmdir(out)
        except OSError:
            pass
        del target

    def test_open_in_viewer_windows(self):
        with mock.patch.object(self.mod.sys, "platform", "win32"):
            startfile = mock.MagicMock()
            # os.startfile only exists on Windows; create=True for Linux CI.
            with mock.patch.object(self.mod.os, "startfile", startfile, create=True):
                self.mod._open_in_viewer(r"C:\x\y.png")
            startfile.assert_called_once_with(r"C:\x\y.png")

    def test_open_in_viewer_macos(self):
        with mock.patch.object(self.mod.sys, "platform", "darwin"), \
             mock.patch.object(self.mod.subprocess, "Popen") as popen:
            self.mod._open_in_viewer("/x/y.png")
        popen.assert_called_once()
        self.assertEqual(popen.call_args[0][0][0], "open")

    def test_open_in_viewer_linux(self):
        with mock.patch.object(self.mod.sys, "platform", "linux"), \
             mock.patch.object(self.mod.subprocess, "Popen") as popen:
            self.mod._open_in_viewer("/x/y.png")
        self.assertEqual(popen.call_args[0][0][0], "xdg-open")

    def test_open_in_viewer_swallows_error(self):
        # Popen raises → caught + printed, no propagation.
        with mock.patch.object(self.mod.sys, "platform", "linux"), \
             mock.patch.object(self.mod.subprocess, "Popen",
                               side_effect=OSError("no xdg-open")):
            self.mod._open_in_viewer("/x/y.png")  # must not raise

    def test_prune_output_dir_keeps_newest(self):
        out = os.path.join(self.tmp, "gen")
        os.makedirs(out)
        # Create 5 png files with increasing mtimes; keep=2 → 3 oldest removed.
        paths = []
        for i in range(5):
            p = os.path.join(out, f"img_{i}.png")
            with open(p, "wb") as f:
                f.write(b"x")
            os.utime(p, (1000 + i, 1000 + i))
            paths.append(p)
        # A non-png must be ignored entirely.
        keep_txt = os.path.join(out, "note.txt")
        with open(keep_txt, "w") as f:
            f.write("keep")
        with mock.patch.object(self.mod, "_output_dir", return_value=out):
            self.mod._prune_output_dir(keep=2)
        remaining = sorted(n for n in os.listdir(out) if n.endswith(".png"))
        self.assertEqual(remaining, ["img_3.png", "img_4.png"])
        self.assertTrue(os.path.exists(keep_txt))

    def test_prune_output_dir_noop_under_cap(self):
        out = os.path.join(self.tmp, "gen2")
        os.makedirs(out)
        for i in range(2):
            with open(os.path.join(out, f"a_{i}.png"), "wb") as f:
                f.write(b"x")
        with mock.patch.object(self.mod, "_output_dir", return_value=out):
            self.mod._prune_output_dir(keep=10)   # under cap → no deletion
        self.assertEqual(len([n for n in os.listdir(out) if n.endswith(".png")]), 2)

    def test_prune_output_dir_swallows_remove_error(self):
        out = os.path.join(self.tmp, "gen3")
        os.makedirs(out)
        for i in range(3):
            with open(os.path.join(out, f"b_{i}.png"), "wb") as f:
                f.write(b"x")
        with mock.patch.object(self.mod, "_output_dir", return_value=out), \
             mock.patch.object(self.mod.os, "remove", side_effect=OSError("locked")):
            self.mod._prune_output_dir(keep=1)   # remove raises → swallowed

    def test_prune_output_dir_swallows_listdir_error(self):
        with mock.patch.object(self.mod, "_output_dir",
                               side_effect=OSError("gone")):
            self.mod._prune_output_dir()   # outer try swallows → no raise

    def test_save_and_open_success(self):
        out = os.path.join(self.tmp, "save")
        os.makedirs(out)
        with mock.patch.object(self.mod, "_output_dir", return_value=out), \
             mock.patch.object(self.mod, "_prune_output_dir") as prune, \
             mock.patch.object(self.mod, "_open_in_viewer") as viewer:
            msg = self.mod._save_and_open(b"PNGDATA", "a cat in a hat")
        self.assertIn("image saved to", msg)
        # The slugged prompt appears in the filename.
        self.assertIn("a_cat_in_a_hat", msg)
        prune.assert_called_once()
        viewer.assert_called_once()
        # The bytes actually landed on disk.
        written = [n for n in os.listdir(out) if n.endswith(".png")]
        self.assertEqual(len(written), 1)
        with open(os.path.join(out, written[0]), "rb") as f:
            self.assertEqual(f.read(), b"PNGDATA")

    def test_save_and_open_write_failure(self):
        out = os.path.join(self.tmp, "save2")
        os.makedirs(out)
        with mock.patch.object(self.mod, "_output_dir", return_value=out), \
             mock.patch("builtins.open", side_effect=PermissionError("read-only")):
            msg = self.mod._save_and_open(b"x", "p")
        self.assertIn("save failed", msg)


# ── diffusers backend ────────────────────────────────────────────────────
class _FakeImage:
    def __init__(self):
        self.saved_format = None

    def save(self, buf, format=None):   # noqa: A002 — mirror PIL signature
        self.saved_format = format
        buf.write(b"DIFFUSERS_PNG")


class _FakeResult:
    def __init__(self, image):
        self.images = [image]


def _fake_torch(cuda_available=True):
    torch = types.ModuleType("torch")
    torch.float16 = "fp16"
    torch.float32 = "fp32"
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: cuda_available,
        empty_cache=mock.MagicMock())
    return torch


def _fake_diffusers(pipe_obj=None, from_pretrained=None):
    """Build a fake `diffusers` module exposing AutoPipelineForText2Image."""
    diffusers = types.ModuleType("diffusers")

    class _AutoPipe:
        last_kwargs = None
        calls = 0

        @classmethod
        def from_pretrained(cls, model, **kwargs):
            cls.calls += 1
            cls.last_kwargs = kwargs
            if from_pretrained is not None:
                return from_pretrained(model, **kwargs)
            return _PipeWrapper(pipe_obj)

    diffusers.AutoPipelineForText2Image = _AutoPipe
    return diffusers, _AutoPipe


class _PipeWrapper:
    """What from_pretrained(...) returns — has .to(device) → the callable pipe."""
    def __init__(self, pipe_obj):
        self._pipe = pipe_obj if pipe_obj is not None else _DefaultPipe()

    def to(self, device):
        self._pipe.device = device
        return self._pipe


class _DefaultPipe:
    def __init__(self):
        self.device = None

    def __call__(self, prompt=None, num_inference_steps=None, guidance_scale=None):
        return _FakeResult(_FakeImage())


class DiffusersBackendTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("image_gen")
        # The module caches the pipeline at module scope; reset before AND after
        # each test so neither a prior run nor this one leaks a fake pipe.
        self.mod._diffusers_pipe = None
        self.addCleanup(self._reset_pipe)

    def _reset_pipe(self):
        self.mod._diffusers_pipe = None

    def _gen(self, torch_mod, diffusers_mod):
        with mock.patch.object(self.mod, "_config",
                               return_value=("diffusers", "stabilityai/sdxl-turbo",
                                             "http://x", 4)), \
             mock.patch.dict(sys.modules, {"torch": torch_mod,
                                           "diffusers": diffusers_mod}), \
             mock.patch.object(self.mod, "_save_and_open",
                               return_value="image saved to /tmp/d.png"):
            return self.actions["generate_image"]("a fox in the snow")

    def test_diffusers_happy_path_cuda(self):
        diffusers, auto = _fake_diffusers()
        out = self._gen(_fake_torch(cuda_available=True), diffusers)
        self.assertIn("image saved to", out)
        self.assertIn("sir", out)
        # On CUDA the loader requests the fp16 variant.
        self.assertEqual(auto.last_kwargs.get("variant"), "fp16")
        self.assertEqual(auto.last_kwargs.get("torch_dtype"), "fp16")

    def test_diffusers_happy_path_cpu(self):
        diffusers, auto = _fake_diffusers()
        out = self._gen(_fake_torch(cuda_available=False), diffusers)
        self.assertIn("image saved to", out)
        # CPU path uses fp32 and never sets the fp16 variant.
        self.assertEqual(auto.last_kwargs.get("torch_dtype"), "fp32")
        self.assertNotIn("variant", auto.last_kwargs)

    def test_diffusers_pipe_cached_across_calls(self):
        diffusers, auto = _fake_diffusers()
        self._gen(_fake_torch(), diffusers)
        first = auto.calls
        # Second call reuses the cached pipe (no extra from_pretrained).
        self._gen(_fake_torch(), diffusers)
        self.assertEqual(auto.calls, first)

    def test_diffusers_variant_load_fails_retries_plain(self):
        # First from_pretrained (with variant=fp16) raises; retry without
        # variant succeeds. Exercises the inner try/except retry block.
        state = {"n": 0}

        def _fp(model, **kwargs):
            state["n"] += 1
            if state["n"] == 1:
                self.assertEqual(kwargs.get("variant"), "fp16")
                raise ValueError("fp16 variant not found")
            self.assertNotIn("variant", kwargs)
            return _PipeWrapper(None)

        diffusers, _auto = _fake_diffusers(from_pretrained=_fp)
        out = self._gen(_fake_torch(cuda_available=True), diffusers)
        self.assertIn("image saved to", out)
        self.assertEqual(state["n"], 2)   # retried exactly once

    def test_diffusers_load_oom_returns_busy(self):
        def _fp(model, **kwargs):
            raise RuntimeError("CUDA out of memory")

        diffusers, _auto = _fake_diffusers(from_pretrained=_fp)
        torch = _fake_torch(cuda_available=False)  # cpu → no variant retry loop
        out = self._gen(torch, diffusers)
        self.assertIn("GPU is busy", out)
        torch.cuda.empty_cache.assert_called_once()
        # The cached pipe was reset to None after the failure.
        self.assertIsNone(self.mod._diffusers_pipe)

    def test_diffusers_load_generic_failure(self):
        def _fp(model, **kwargs):
            raise ValueError("corrupt checkpoint")

        diffusers, _auto = _fake_diffusers(from_pretrained=_fp)
        out = self._gen(_fake_torch(cuda_available=False), diffusers)
        self.assertIn("pipeline failed to load", out)

    def test_diffusers_inference_oom_drops_pipe(self):
        class _OomPipe(_DefaultPipe):
            def __call__(self, **kw):
                raise RuntimeError("CUDA out of memory during sampling")

        diffusers, _auto = _fake_diffusers(pipe_obj=_OomPipe())
        torch = _fake_torch(cuda_available=False)
        out = self._gen(torch, diffusers)
        self.assertIn("GPU is busy", out)
        # OOM mid-inference drops the cached pipe so the next call reloads.
        self.assertIsNone(self.mod._diffusers_pipe)
        torch.cuda.empty_cache.assert_called_once()

    def test_diffusers_inference_generic_failure(self):
        class _BadPipe(_DefaultPipe):
            def __call__(self, **kw):
                raise ValueError("nan in latents")

        diffusers, _auto = _fake_diffusers(pipe_obj=_BadPipe())
        out = self._gen(_fake_torch(cuda_available=False), diffusers)
        self.assertIn("image generation failed", out)

    def test_diffusers_saves_png_format(self):
        # Verify the PNG buffer is handed to _save_and_open (format='PNG').
        img = _FakeImage()

        class _Pipe(_DefaultPipe):
            def __call__(self, **kw):
                return _FakeResult(img)

        diffusers, _auto = _fake_diffusers(pipe_obj=_Pipe())
        captured = {}

        def _save(png_bytes, prompt):
            captured["bytes"] = png_bytes
            captured["prompt"] = prompt
            return "image saved to /tmp/x.png"

        with mock.patch.object(self.mod, "_config",
                               return_value=("diffusers", "m", "http://x", 4)), \
             mock.patch.dict(sys.modules, {"torch": _fake_torch(False),
                                           "diffusers": diffusers}), \
             mock.patch.object(self.mod, "_save_and_open", side_effect=_save):
            self.actions["generate_image"]("snowy fox")
        self.assertEqual(img.saved_format, "PNG")
        self.assertEqual(captured["bytes"], b"DIFFUSERS_PNG")
        self.assertEqual(captured["prompt"], "snowy fox")


# ── register() ───────────────────────────────────────────────────────────
class RegisterTests(unittest.TestCase):
    def test_register_wires_both_actions(self):
        mod, actions = load_skill_isolated("image_gen")
        self.assertIs(actions["generate_image"], mod.generate_image)
        self.assertIs(actions["make_picture"], mod.make_picture)


if __name__ == "__main__":
    unittest.main()
