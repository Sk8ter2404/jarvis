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


if __name__ == "__main__":
    unittest.main()
