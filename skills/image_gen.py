"""
Local image generation skill for JARVIS.

Generates images locally on the workshop GPU (the RTX 3090 has 24 GB of VRAM,
which is plenty for SDXL-Turbo) via either ComfyUI's HTTP API or HuggingFace
`diffusers` running the SDXL-Turbo pipeline. Generated images land in
`./screenshots/JARVIS_generated/` and open in the system's default viewer when
the render finishes.

Actions added:
  generate_image  <prompt>   — render a new image from a text prompt
  make_picture    <prompt>   — alias of generate_image (matches the
                                'make me a picture of X' voice phrasing)

Voice phrasing the LLM is expected to map onto these actions:
  "generate an image of a Mars colony at sunset"
                            → generate_image, a Mars colony at sunset
  "make me a picture of a cat in a spacesuit"
                            → make_picture, a cat in a spacesuit

Config (resolved in this order: env var → bobert_companion attr → built-in
default):
  IMAGE_GEN_BACKEND      — 'comfyui' | 'diffusers' | 'off'   (default 'off')
  IMAGE_GEN_MODEL        — diffusers repo id (default 'stabilityai/sdxl-turbo')
                           OR comfyui checkpoint filename (default
                           'sd_xl_turbo_1.0_fp16.safetensors')
  IMAGE_GEN_COMFYUI_URL  — http base url (default 'http://localhost:8188')
  IMAGE_GEN_STEPS        — sampling steps (default 4, optimal for sdxl-turbo)

The default backend is 'off' so the actions register cleanly but refuse to
allocate VRAM until the user explicitly opts in by exporting
IMAGE_GEN_BACKEND or flipping the constant in bobert_companion.py.

Optional dependencies — only need to be installed for the backend you pick:
  diffusers backend → pip install diffusers torch transformers accelerate
  comfyui backend  → run ComfyUI separately, no Python deps required here
"""
from __future__ import annotations

import io
import os
import sys
import time
import uuid
import threading
import subprocess

_DEFAULT_COMFYUI_URL    = "http://localhost:8188"
_DEFAULT_DIFFUSERS_MODEL = "stabilityai/sdxl-turbo"
_DEFAULT_COMFYUI_CKPT   = "sd_xl_turbo_1.0_fp16.safetensors"
_DEFAULT_STEPS          = 4
# SDXL-Turbo is a distilled model trained to ignore the CFG scale — 1.0 is the
# canonical value. Bumping CFG just slows generation without improving quality.
_DEFAULT_CFG            = 1.0
_OUTPUT_SUBDIR          = "JARVIS_generated"
# Network deadline for the comfyui poll loop. SDXL-Turbo finishes in 1-2 s on
# the 3090; 120 s is the safety net for first-call model load (~10 s) plus a
# busy queue.
_COMFYUI_TIMEOUT        = 120.0

# Lazy-loaded diffusers pipeline. Cached at module scope so subsequent calls
# skip the ~6 GB checkpoint load + the VRAM swap-in.
_diffusers_pipe = None
_diffusers_lock = threading.Lock()


def _is_cuda_oom(exc: Exception) -> bool:
    """True when an exception looks like a CUDA out-of-memory error. Matched
    by message (not type) so it works without importing torch here."""
    msg = f"{type(exc).__name__}: {exc}".lower()
    return "out of memory" in msg or "cuda" in msg


def _free_cuda_cache() -> None:
    """Best-effort release of cached VRAM after an OOM. Guards the torch
    import so a missing/broken torch never breaks the error path."""
    try:
        import torch  # type: ignore[import-not-found]
        torch.cuda.empty_cache()
    except Exception:
        pass


def _bobert():
    """Resolve the bobert_companion module so we can read config constants
    set at the top of bobert_companion.py without forcing a circular import."""
    return sys.modules.get("__main__") or sys.modules.get("bobert_companion")


def _resolve(env_name: str, attr_name: str, default):
    """env var > bobert constant > built-in default."""
    val = os.environ.get(env_name)
    if val is not None and val != "":
        return val
    b = _bobert()
    if b is not None and hasattr(b, attr_name):
        v = getattr(b, attr_name)
        if v is not None and v != "":
            return v
    return default


def _config() -> tuple[str, str, str, int]:
    backend = str(_resolve("IMAGE_GEN_BACKEND", "IMAGE_GEN_BACKEND", "off")).strip().lower()
    if backend not in ("comfyui", "diffusers", "off"):
        backend = "off"
    default_model = (_DEFAULT_DIFFUSERS_MODEL if backend == "diffusers"
                     else _DEFAULT_COMFYUI_CKPT)
    model = str(_resolve("IMAGE_GEN_MODEL", "IMAGE_GEN_MODEL", default_model)).strip()
    url = str(_resolve("IMAGE_GEN_COMFYUI_URL", "IMAGE_GEN_COMFYUI_URL",
                       _DEFAULT_COMFYUI_URL)).rstrip("/")
    try:
        steps = int(_resolve("IMAGE_GEN_STEPS", "IMAGE_GEN_STEPS", _DEFAULT_STEPS))
    except (TypeError, ValueError):
        steps = _DEFAULT_STEPS
    return backend, model, url, max(1, steps)


def _output_dir() -> str:
    # skills/image_gen.py → parent dir is the JARVIS root.
    here = os.path.dirname(os.path.abspath(__file__))
    parent = os.path.dirname(here)
    out = os.path.join(parent, "screenshots", _OUTPUT_SUBDIR)
    os.makedirs(out, exist_ok=True)
    return out


def _slug(prompt: str, limit: int = 40) -> str:
    import re
    s = re.sub(r"[^A-Za-z0-9]+", "_", prompt).strip("_").lower()
    return s[:limit] or "image"


def _open_in_viewer(path: str) -> None:
    try:
        if sys.platform == "win32":
            os.startfile(path)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path], close_fds=True)
        else:
            subprocess.Popen(["xdg-open", path], close_fds=True)
    except Exception as e:
        print(f"  [image-gen] couldn't open viewer for {path}: {e}")


_MAX_GENERATED_IMAGES = 50  # cap the output dir so renders don't grow unbounded


def _prune_output_dir(keep: int = _MAX_GENERATED_IMAGES) -> None:
    """Keep only the `keep` newest *.png in the output dir, deleting the
    oldest by mtime. Best-effort + exception-safe: a prune failure must
    never sink an otherwise-successful render."""
    try:
        out = _output_dir()
        pngs = [os.path.join(out, n) for n in os.listdir(out)
                if n.lower().endswith(".png")]
        if len(pngs) <= keep:
            return
        pngs.sort(key=lambda p: os.path.getmtime(p))
        for old in pngs[:len(pngs) - keep]:
            try:
                os.remove(old)
            except OSError:
                pass
    except Exception:
        pass


def _save_and_open(png_bytes: bytes, prompt: str) -> str:
    fname = f"{time.strftime('%Y%m%d_%H%M%S')}_{_slug(prompt)}.png"
    path = os.path.join(_output_dir(), fname)
    try:
        with open(path, "wb") as f:
            f.write(png_bytes)
    except Exception as e:
        return f"image generated but save failed: {e}"
    _prune_output_dir()
    _open_in_viewer(path)
    return f"image saved to {path}"


# ── Diffusers backend ───────────────────────────────────────────────────
def _generate_diffusers(prompt: str, model: str, steps: int) -> str:
    try:
        import torch
        from diffusers import AutoPipelineForText2Image
    except ImportError as e:
        return ("diffusers / torch isn't installed, sir — "
                f"pip install diffusers torch transformers accelerate ({e})")

    global _diffusers_pipe
    with _diffusers_lock:
        if _diffusers_pipe is None:
            print(f"  [image-gen] loading diffusers pipeline `{model}` "
                  "(first run downloads ~6 GB)…", flush=True)
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                dtype = torch.float16 if device == "cuda" else torch.float32
                kwargs = {"torch_dtype": dtype}
                # `variant='fp16'` lets us skip the fp32 weights download on
                # repos that ship both — but not every model has the variant,
                # so retry plain on failure.
                if device == "cuda":
                    kwargs["variant"] = "fp16"
                try:
                    _diffusers_pipe = AutoPipelineForText2Image.from_pretrained(
                        model, **kwargs
                    ).to(device)
                except Exception:
                    kwargs.pop("variant", None)
                    _diffusers_pipe = AutoPipelineForText2Image.from_pretrained(
                        model, **kwargs
                    ).to(device)
            except Exception as e:
                _diffusers_pipe = None
                # The 3090 is shared with whisper + qwen, so the .to('cuda')
                # move can lose the VRAM race. Free the half-allocated cache
                # and tell the user to retry rather than crashing.
                if _is_cuda_oom(e):
                    _free_cuda_cache()
                    return ("GPU is busy, sir — try again in a moment.")
                return f"diffusers pipeline failed to load: {e}"

    try:
        result = _diffusers_pipe(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=_DEFAULT_CFG,
        )
        image = result.images[0]
    except Exception as e:
        # OOM mid-inference leaves VRAM fragmented and the cached pipe in a
        # bad state — drop it, free the cache, and ask the user to retry.
        # (_diffusers_pipe is already declared global at the top of this fn.)
        if _is_cuda_oom(e):
            with _diffusers_lock:
                _diffusers_pipe = None
            _free_cuda_cache()
            return ("GPU is busy, sir — try again in a moment.")
        return f"diffusers image generation failed: {e}"

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return _save_and_open(buf.getvalue(), prompt)


# ── ComfyUI backend ─────────────────────────────────────────────────────
def _comfyui_workflow(prompt: str, ckpt: str, steps: int, seed: int) -> dict:
    """Build the minimal SDXL-Turbo workflow ComfyUI's /prompt endpoint
    expects (the 'API format' graph: a dict keyed by stringified node IDs).

    Graph: CheckpointLoader → CLIPTextEncode (pos + empty neg) → EmptyLatent
    → KSampler → VAEDecode → SaveImage."""
    return {
        "3": {"class_type": "KSampler", "inputs": {
            "seed": seed,
            "steps": steps,
            "cfg": _DEFAULT_CFG,
            "sampler_name": "euler_ancestral",
            "scheduler": "normal",
            "denoise": 1.0,
            "model": ["4", 0],
            "positive": ["6", 0],
            "negative": ["7", 0],
            "latent_image": ["5", 0],
        }},
        "4": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": ckpt}},
        "5": {"class_type": "EmptyLatentImage",
              "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
        "6": {"class_type": "CLIPTextEncode",
              "inputs": {"text": prompt, "clip": ["4", 1]}},
        "7": {"class_type": "CLIPTextEncode",
              "inputs": {"text": "", "clip": ["4", 1]}},
        "8": {"class_type": "VAEDecode",
              "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
        "9": {"class_type": "SaveImage",
              "inputs": {"images": ["8", 0], "filename_prefix": "jarvis"}},
    }


def _generate_comfyui(prompt: str, ckpt: str, url: str, steps: int) -> str:
    try:
        import requests
    except ImportError:
        return "requests isn't installed, sir — pip install requests"

    # Probe that ComfyUI is up before submitting a job so we give a clear
    # error rather than a cryptic post failure.
    try:
        r = requests.get(f"{url}/system_stats", timeout=3)
        if not r.ok:
            return (f"ComfyUI returned HTTP {r.status_code} at {url}, sir — "
                    "is ComfyUI running?")
    except Exception as e:
        return (f"I couldn't reach ComfyUI at {url}, sir — "
                f"start ComfyUI or set IMAGE_GEN_COMFYUI_URL ({e})")

    import random
    seed = random.randint(1, 2**31 - 1)
    workflow = _comfyui_workflow(prompt, ckpt, steps, seed)
    client_id = uuid.uuid4().hex

    try:
        r = requests.post(f"{url}/prompt",
                          json={"prompt": workflow, "client_id": client_id},
                          timeout=10)
        if not r.ok:
            return (f"ComfyUI rejected the workflow: HTTP {r.status_code} — "
                    f"{r.text[:200]}")
        prompt_id = r.json().get("prompt_id")
        if not prompt_id:
            return "ComfyUI didn't return a prompt_id, sir."
    except Exception as e:
        return f"ComfyUI submit failed: {e}"

    deadline = time.time() + _COMFYUI_TIMEOUT
    history = None
    while time.time() < deadline:
        try:
            h = requests.get(f"{url}/history/{prompt_id}", timeout=5)
            if h.ok:
                data = h.json()
                if prompt_id in data:
                    history = data[prompt_id]
                    break
        except Exception:
            pass
        time.sleep(0.4)

    if history is None:
        return (f"ComfyUI didn't finish the job within "
                f"{int(_COMFYUI_TIMEOUT)} seconds, sir.")

    # Walk the outputs dict — only the SaveImage node will populate `images`
    # but the index isn't fixed, so iterate.
    outputs = history.get("outputs") or {}
    images_info: list[dict] = []
    for node_out in outputs.values():
        for img in node_out.get("images") or []:
            images_info.append(img)
    if not images_info:
        return "ComfyUI finished but produced no images, sir."

    img = images_info[0]
    try:
        v = requests.get(f"{url}/view", params={
            "filename": img["filename"],
            "subfolder": img.get("subfolder", ""),
            "type": img.get("type", "output"),
        }, timeout=30)
        if not v.ok:
            return ("ComfyUI couldn't return the generated image: "
                    f"HTTP {v.status_code}")
        return _save_and_open(v.content, prompt)
    except Exception as e:
        return f"ComfyUI image fetch failed: {e}"


# ── public actions ──────────────────────────────────────────────────────
def generate_image(prompt: str = "") -> str:
    prompt = (prompt or "").strip().strip("'\"")
    if not prompt:
        return "format: generate_image, <description of the image>"

    backend, model, url, steps = _config()
    if backend == "off":
        return ("local image generation is off, sir — "
                "set IMAGE_GEN_BACKEND to 'comfyui' or 'diffusers' to enable it")

    print(f"  [image-gen] generating via {backend} ({steps} steps): "
          f"{prompt[:80]}", flush=True)
    t0 = time.time()
    if backend == "diffusers":
        result = _generate_diffusers(prompt, model, steps)
    else:
        result = _generate_comfyui(prompt, model, url, steps)
    dt = time.time() - t0
    if result.startswith("image saved to "):
        result = f"{result} ({dt:.1f}s), sir."
    return result


# Alias so the LLM doesn't have to remember a single canonical name —
# matches the 'make me a picture of X' voice phrasing exactly.
def make_picture(prompt: str = "") -> str:
    return generate_image(prompt)


def register(actions: dict):
    actions["generate_image"] = generate_image
    actions["make_picture"]   = make_picture
