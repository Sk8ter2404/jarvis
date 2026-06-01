#!/usr/bin/env python3
"""
Face-detect benchmark harness — CPU vs. CUDA Haar cascade.

Background: bobert_companion._detect_face runs a 4-pass Haar cascade chain
(frontal-strict → frontal-loose → profile → profile-mirrored) on every webcam
frame the face-tracking thread grabs. jarvis_todo 2026-05-29 13:34 asked whether
porting that to cv2.cuda.CascadeClassifier on the 3090 would be worth the
opencv-contrib-CUDA build pain. This harness lets you re-evaluate at any time:

    python tools/face_detect_bench.py                  # synthetic frame, 1280x720
    python tools/face_detect_bench.py --image foo.jpg  # real frame from disk
    python tools/face_detect_bench.py --camera 0       # grab N frames from a webcam
    python tools/face_detect_bench.py --width 640 --height 480
    python tools/face_detect_bench.py --iters 500

The script reports per-pass CPU timing and, IF a CUDA-enabled OpenCV build is
present (cv2.cuda.getCudaEnabledDeviceCount() > 0 AND
cv2.cuda_CascadeClassifier is importable), the matching CUDA timing — including
the HtoD/DtoH transfer cost, which is what kills the win on small frames.

Exit code is 0 unless a hard failure happens (missing cascade XML, bad camera
index). It does NOT exit non-zero when CUDA is absent — that's the expected
state for the default pip-installed opencv-python.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np


FRONTAL_XML = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
PROFILE_XML = cv2.data.haarcascades + "haarcascade_profileface.xml"


def _make_synthetic_frame(w: int, h: int) -> np.ndarray:
    frame = np.full((h, w, 3), 128, dtype=np.uint8)
    cv2.circle(frame, (w // 2, h // 2), min(w, h) // 9, (200, 180, 170), -1)
    return frame


def _load_frame(path: str) -> np.ndarray:
    frame = cv2.imread(path)
    if frame is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return frame


def _grab_camera_frame(index: int, w: int, h: int) -> np.ndarray:
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"could not open camera index {index}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"camera {index} opened but yielded no frame")
    return frame


def _summarize(samples: list[float], label: str) -> None:
    if not samples:
        print(f"  {label}: (no samples)")
        return
    mean = statistics.mean(samples)
    p50 = statistics.median(samples)
    p95 = statistics.quantiles(samples, n=20)[-1] if len(samples) >= 20 else max(samples)
    print(f"  {label:38} mean {mean:6.2f} ms   p50 {p50:6.2f} ms   p95 {p95:6.2f} ms")


def bench_cpu(frame: np.ndarray, iters: int) -> dict[str, list[float]]:
    frontal = cv2.CascadeClassifier(FRONTAL_XML)
    profile = cv2.CascadeClassifier(PROFILE_XML)
    if frontal.empty():
        raise RuntimeError(f"frontal cascade XML failed to load: {FRONTAL_XML}")
    if profile.empty():
        raise RuntimeError(f"profile cascade XML failed to load: {PROFILE_XML}")

    samples: dict[str, list[float]] = {
        "preprocess (BGR2GRAY + equalizeHist)": [],
        "strict frontal pass only": [],
        "loose frontal pass only": [],
        "profile pass only": [],
        "profile-mirrored pass only": [],
        "full 4-pass chain (worst case)": [],
    }

    # Warm-up: cascade XML lazy-parses on first detectMultiScale.
    gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    for _ in range(5):
        frontal.detectMultiScale(gray, 1.05, 4, minSize=(40, 40))
        profile.detectMultiScale(gray, 1.05, 4, minSize=(40, 40))

    for _ in range(iters):
        t0 = time.perf_counter()
        gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        samples["preprocess (BGR2GRAY + equalizeHist)"].append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        frontal.detectMultiScale(gray, 1.05, 4, minSize=(40, 40))
        samples["strict frontal pass only"].append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        frontal.detectMultiScale(gray, 1.05, 3, minSize=(60, 60))
        samples["loose frontal pass only"].append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        profile.detectMultiScale(gray, 1.05, 4, minSize=(40, 40))
        samples["profile pass only"].append((time.perf_counter() - t0) * 1000)

        mirror = cv2.flip(gray, 1)
        t0 = time.perf_counter()
        profile.detectMultiScale(mirror, 1.05, 4, minSize=(40, 40))
        samples["profile-mirrored pass only"].append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        g = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        f1 = frontal.detectMultiScale(g, 1.05, 4, minSize=(40, 40))
        if len(f1) == 0:
            f1 = frontal.detectMultiScale(g, 1.05, 3, minSize=(60, 60))
        if len(f1) == 0:
            f1 = profile.detectMultiScale(g, 1.05, 4, minSize=(40, 40))
            if len(f1) == 0:
                profile.detectMultiScale(cv2.flip(g, 1), 1.05, 4, minSize=(40, 40))
        samples["full 4-pass chain (worst case)"].append((time.perf_counter() - t0) * 1000)

    return samples


def _cuda_available() -> tuple[bool, str]:
    try:
        n = cv2.cuda.getCudaEnabledDeviceCount()
    except Exception as e:
        return False, f"cv2.cuda namespace missing ({type(e).__name__}: {e})"
    if n <= 0:
        return False, "cv2.cuda present but 0 CUDA devices (OpenCV not built with CUDA)"
    if not hasattr(cv2, "cuda_CascadeClassifier"):
        return False, f"{n} CUDA device(s) but cv2.cuda_CascadeClassifier missing — opencv-contrib CUDA build needed"
    return True, f"{n} CUDA device(s), cv2.cuda_CascadeClassifier present"


def bench_cuda(frame: np.ndarray, iters: int) -> dict[str, list[float]]:
    cls = getattr(cv2, "cuda_CascadeClassifier", None)
    if cls is None:
        return {}
    create = getattr(cls, "create", None)
    if create is None:
        return {}
    try:
        frontal_cuda = create(FRONTAL_XML)
        profile_cuda = create(PROFILE_XML)
    except Exception as e:
        print(f"  [cuda] failed to construct CUDA cascades: {type(e).__name__}: {e}")
        return {}

    samples: dict[str, list[float]] = {
        "CUDA upload (HtoD) only": [],
        "CUDA preprocess (gray+equalize on GPU)": [],
        "CUDA strict frontal only": [],
        "CUDA download (DtoH) of result": [],
        "CUDA full 4-pass chain (worst case)": [],
    }

    g_frame = cv2.cuda_GpuMat()
    for _ in range(5):
        g_frame.upload(frame)
        g_gray = cv2.cuda.cvtColor(g_frame, cv2.COLOR_BGR2GRAY)
        g_gray = cv2.cuda.equalizeHist(g_gray)
        frontal_cuda.detectMultiScale(g_gray).download()

    for _ in range(iters):
        t0 = time.perf_counter()
        g_frame.upload(frame)
        samples["CUDA upload (HtoD) only"].append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        g_gray = cv2.cuda.cvtColor(g_frame, cv2.COLOR_BGR2GRAY)
        g_gray = cv2.cuda.equalizeHist(g_gray)
        samples["CUDA preprocess (gray+equalize on GPU)"].append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        objs = frontal_cuda.detectMultiScale(g_gray)
        samples["CUDA strict frontal only"].append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        objs.download()
        samples["CUDA download (DtoH) of result"].append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        g_frame.upload(frame)
        g = cv2.cuda.cvtColor(g_frame, cv2.COLOR_BGR2GRAY)
        g = cv2.cuda.equalizeHist(g)
        r = frontal_cuda.detectMultiScale(g).download()
        if r is None or len(r) == 0:
            # Profile cascade pass
            r = profile_cuda.detectMultiScale(g).download()
            if r is None or len(r) == 0:
                g_mirror = cv2.cuda.flip(g, 1)
                profile_cuda.detectMultiScale(g_mirror).download()
        samples["CUDA full 4-pass chain (worst case)"].append((time.perf_counter() - t0) * 1000)

    return samples


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--image", help="path to a frame to use instead of synthetic")
    ap.add_argument("--camera", type=int, help="grab a single frame from this camera index")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--iters", type=int, default=200)
    args = ap.parse_args(argv)

    print(f"OpenCV {cv2.__version__}")
    cuda_ok, cuda_msg = _cuda_available()
    print(f"CUDA: {cuda_msg}")

    if args.image:
        frame = _load_frame(args.image)
    elif args.camera is not None:
        frame = _grab_camera_frame(args.camera, args.width, args.height)
    else:
        frame = _make_synthetic_frame(args.width, args.height)

    h, w = frame.shape[:2]
    print(f"Frame: {w}x{h}, {args.iters} iterations\n")

    print("CPU (cv2.CascadeClassifier):")
    for label, samples in bench_cpu(frame, args.iters).items():
        _summarize(samples, label)

    if cuda_ok:
        print("\nCUDA (cv2.cuda_CascadeClassifier):")
        cuda_samples = bench_cuda(frame, args.iters)
        if not cuda_samples:
            print("  (CUDA cascades could not be constructed — see error above)")
        else:
            for label, samples in cuda_samples.items():
                _summarize(samples, label)
    else:
        print("\n(skipping CUDA — install an opencv-contrib build with CUDA to enable)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
