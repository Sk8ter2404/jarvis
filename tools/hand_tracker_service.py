"""JARVIS hand tracker service.

Runs with the SEPARATE tracker venv's Python (mediapipe + websockets + their
own OpenCV 5), never with JARVIS's interpreter:

    C:\\JARVIS-models\\handtrack\\venv\\Scripts\\python.exe tools\\hand_tracker_service.py

It is started and stopped by JARVIS (skills/hand_tracking.py ->
core/hand_tracker_launcher.py), only while hand tracking is turned on.

DATA FLOW
  JARVIS camera loop --(shared memory "jarvis_handtrack_frame", 640x480 BGR,
  core/handtrack_frames.py)--> this service --(MediaPipe HandLandmarker,
  VIDEO mode, 2 hands)--> tools/handtrack_gestures.py (states + events)
  --(ws://127.0.0.1:8793)--> the holo page / desktop trigger.

The frame is MIRRORED (cv2.flip) before detection, so MediaPipe's handedness
labels are the owner's real hands and every coordinate is already in display
(mirror) orientation, as the protocol asks.

PROTOCOL  (JSON text frames, ~30/s while frames arrive)
  {"type":"hands","t":ms,"seq":n,"w":640,"h":480,"hands":[{"id":0,
    "side":"Left"|"Right","score":0-1,"lm":[[x,y,z]*21 normalised, mirrored],
    "world":[[x,y,z]*21 metres],"state":"open|fist|pinch|point|unknown",
    "pinch":0-1,"palm":[x,y],"palm_facing":bool,"size":hs/height,"ok":bool}]}
      (size / ok, 2026-10-20: the hand's size over the frame height, and whether
      it may fire trigger events - tools/handtrack_gestures.py RECOGNITION)
  {"type":"gesture","name":...,"side":"Left|Right|both","pos":[x,y],"value"?}
      names: palm_hold pinch_start pinch_end pinch_hold fist_grab fist_release
             point_flick explode clap   (see tools/handtrack_gestures.py)
  Client -> service: {"type":"subscribe","frames":true|false,"depth":true|false}
  (each key optional: an absent key leaves that subscription as it was).
  Clients subscribed to frames also receive binary WebSocket messages: JPEG
  (quality 70) of the mirrored frame, at most 30/s (starts FF D8).
  KINECT DEPTH (Phase 2, only when JARVIS publishes a depth plane, see
  core/handtrack_depth.py): clients subscribed to depth receive one binary
  message per new plane, magic "JVDP", 24-byte header + uint16 mm plane,
  mirrored like the picture (layout in core/handtrack_depth.py). And while a
  fresh plane exists every "hands" message gets "depth":true and each hand a
  "z": the palm's median depth in metres (null when the palm has no valid
  depth: out of 0.5-4.5 m range or shadowed). No plane, no "z" keys at all.

LIFECYCLE
  * Binds 127.0.0.1 only (--host must be a loopback address) and accepts
    browser connections only from a localhost origin.
  * Writes the heartbeat file (data/.handtrack_alive, a small JSON status)
    every second: JARVIS's frame publisher runs only while it is fresh.
  * Exits (and removes the heartbeat) when JARVIS goes away: the
    --jarvis-pid process is gone, or the --jarvis-alive file stops being
    touched for --stale-s seconds. It never outlives JARVIS.
  * Never moves the cursor, sends OS input or opens a window.
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import re
import signal
import sys
import threading
import time
from typing import Optional

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import handtrack_gestures as hg  # noqa: E402

DEFAULT_PORT = 8793
DEFAULT_MODEL = r"C:\JARVIS-models\handtrack\hand_landmarker.task"
LOOPBACK = ("127.0.0.1", "::1", "localhost")
ORIGIN_RE = re.compile(r"^https?://(127\.0\.0\.1|localhost|\[::1\])(:\d+)?$")
JPEG_QUALITY = 70


def log(msg: str) -> None:
    print(f"[hand-tracker] {msg}", flush=True)


def _load_frames_module():
    """core/handtrack_frames.py by path (the layout lives there only)."""
    path = os.path.join(ROOT, "core", "handtrack_frames.py")
    spec = importlib.util.spec_from_file_location("handtrack_frames_reader", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["handtrack_frames_reader"] = mod
    spec.loader.exec_module(mod)
    return mod


def _parent_watch():
    """core/parent_watch.py by path: the repo's authoritative liveness check
    (WaitForSingleObject + exit code; a bare GetExitCodeProcess misses
    unreaped / terminating-forever corpses)."""
    path = os.path.join(ROOT, "core", "parent_watch.py")
    try:
        spec = importlib.util.spec_from_file_location("parent_watch_reader", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    except Exception:
        return None


_PARENT_WATCH = _parent_watch()


def _load_depth_module():
    """core/handtrack_depth.py by path: the depth segment + wire layout and the
    median sampler live there only. None when it cannot be loaded (the tracker
    then simply has no depth)."""
    path = os.path.join(ROOT, "core", "handtrack_depth.py")
    try:
        spec = importlib.util.spec_from_file_location("handtrack_depth_reader", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["handtrack_depth_reader"] = mod
        spec.loader.exec_module(mod)
        return mod
    except Exception:
        return None


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if _PARENT_WATCH is not None:
        try:
            return bool(_PARENT_WATCH.parent_is_alive(pid))
        except Exception:
            pass
    if os.name == "nt":
        import ctypes
        k32 = ctypes.windll.kernel32
        h = k32.OpenProcess(0x1000, False, pid)        # QUERY_LIMITED_INFORMATION
        if not h:
            return False
        try:
            code = ctypes.c_ulong(0)
            if not k32.GetExitCodeProcess(h, ctypes.byref(code)):
                return True
            return code.value == 259                  # STILL_ACTIVE
        finally:
            k32.CloseHandle(h)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def file_age(path: Optional[str]) -> Optional[float]:
    if not path:
        return None
    try:
        return max(0.0, time.time() - os.stat(path).st_mtime)
    except OSError:
        return None


def hands_message(ts_ms: int, seq: int, w: int, h: int, hands: list,
                  depth: bool = False) -> dict:
    msg = {"type": "hands", "t": ts_ms, "seq": seq, "w": w, "h": h,
           "hands": hands}
    if depth:
        msg["depth"] = True            # a fresh depth plane exists: hands carry "z"
    return msg


def hand_payload(info: dict, lm, world) -> dict:
    """One entry of the "hands" message: the engine's info + landmarks."""
    return {
        "id": info["id"], "side": info["side"],
        "score": round(float(info["score"]), 3),
        "lm": [[round(p[0], 4), round(p[1], 4), round(p[2], 4)] for p in lm],
        "world": [[round(p[0], 4), round(p[1], 4), round(p[2], 4)] for p in world],
        "state": info["state"], "pinch": info["pinch"],
        "palm": info["palm"], "palm_facing": info["palm_facing"],
        "size": info.get("size", 0.0), "ok": bool(info.get("ok", False)),
    }


class Hub:
    """The websocket side: one asyncio loop in a thread, per-client queues."""

    def __init__(self, host: str, port: int):
        self.host, self.port = host, port
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.clients: dict = {}          # ws -> {"q": Queue, "frames": bool, "depth": bool}
        self._ready = threading.Event()
        self._stop: Optional[asyncio.Event] = None
        self.error: Optional[str] = None
        self.last_hands_text: Optional[str] = None
        self._thread = threading.Thread(target=self._run, name="ht-ws", daemon=True)

    def start(self, timeout: float = 5.0) -> bool:
        self._thread.start()
        self._ready.wait(timeout)
        return self.error is None and self._ready.is_set()

    def _run(self) -> None:
        try:
            asyncio.run(self._main())
        except Exception as e:                       # bind failure etc.
            self.error = f"{type(e).__name__}: {e}"
            log(f"websocket server failed: {self.error}")
            self._ready.set()

    async def _main(self) -> None:
        from websockets.asyncio.server import serve
        self.loop = asyncio.get_running_loop()
        self._stop = asyncio.Event()
        async with serve(self._handler, self.host, self.port,
                         origins=[None, ORIGIN_RE], max_size=4096,
                         ping_interval=20, ping_timeout=20):
            log(f"serving ws://{self.host}:{self.port}")
            self._ready.set()
            await self._stop.wait()

    def stop(self) -> None:
        if self.loop is not None and self._stop is not None:
            self.loop.call_soon_threadsafe(self._stop.set)
        self._thread.join(timeout=3.0)

    async def _handler(self, ws) -> None:
        state = {"q": asyncio.Queue(maxsize=8), "frames": False, "depth": False}
        self.clients[ws] = state
        log(f"client connected ({len(self.clients)})")
        if self.last_hands_text:
            state["q"].put_nowait(self.last_hands_text)
        sender = asyncio.create_task(self._sender(ws, state["q"]))
        try:
            async for msg in ws:
                if isinstance(msg, bytes):
                    continue
                try:
                    data = json.loads(msg)
                except Exception:
                    continue
                if isinstance(data, dict) and data.get("type") == "subscribe":
                    self.apply_subscribe(state, data)
        except Exception:
            pass
        finally:
            sender.cancel()
            self.clients.pop(ws, None)
            log(f"client left ({len(self.clients)})")

    @staticmethod
    def apply_subscribe(state: dict, data: dict) -> None:
        """{"type":"subscribe","frames":bool,"depth":bool}: each key is optional.
        An absent key leaves that subscription as it was, except that a bare
        subscribe (neither key) still means frames off, as it always did."""
        if "frames" in data or "depth" not in data:
            state["frames"] = bool(data.get("frames"))
        if "depth" in data:
            state["depth"] = bool(data.get("depth"))

    @staticmethod
    async def _sender(ws, q) -> None:
        try:
            while True:
                item = await q.get()
                await ws.send(item)
        except Exception:
            return

    # called from the tracker thread
    def wants_frames(self) -> bool:
        return any(c["frames"] for c in list(self.clients.values()))

    def wants_depth(self) -> bool:
        return any(c.get("depth") for c in list(self.clients.values()))

    def client_count(self) -> int:
        return len(self.clients)

    def broadcast(self, text: Optional[str] = None, jpeg: Optional[bytes] = None,
                  depth: Optional[bytes] = None) -> None:
        if self.loop is None:
            return
        if text is not None and text.startswith('{"type":"hands"'):
            self.last_hands_text = text
        self.loop.call_soon_threadsafe(self._fanout, text, jpeg, depth)

    def _fanout(self, text, jpeg, depth=None) -> None:
        for state in list(self.clients.values()):
            for item, only in ((text, None), (jpeg, "frames"), (depth, "depth")):
                if item is None or (only is not None and not state.get(only)):
                    continue
                q = state["q"]
                if q.full():
                    try:
                        q.get_nowait()               # drop the oldest, keep live
                    except Exception:
                        pass
                try:
                    q.put_nowait(item)
                except Exception:
                    pass


class Tracker:
    def __init__(self, args):
        self.args = args
        self.frames = _load_frames_module()
        self.depthmod = _load_depth_module()
        self.engine = hg.GestureEngine(invert_palm=args.invert_palm)
        self.hub = Hub(args.host, args.port)
        self.stop = threading.Event()
        self.shm = None
        self.landmarker = None
        self.started = time.time()
        self.fps = 0.0
        self._last_wh = (self.frames.WIDTH, self.frames.HEIGHT)
        self.processed = 0
        self.hands_now = 0
        self._last_beat = 0.0
        self._jarvis_seen = False
        self._last_exit_check = 0.0
        self._last_attach_try = 0.0
        self._last_ts_ms = 0
        self._last_frame_at = 0.0
        self._empty_sent = True
        self._fps_t0 = time.monotonic()
        self._fps_n = 0
        self._cv2 = None
        self._mp = None
        # Kinect depth (Phase 2): the plane is mirrored once per new plane so it
        # lines up with the mirrored picture and landmarks.
        self.depth_shm = None
        self._last_depth_attach_try = 0.0
        self._depth_seq = 0
        self._depth_plane = None          # mirrored (h, w) uint16 mm, or None
        self._depth_color_seq = 0
        self._depth_ts_ms = 0
        self._depth_relayed = 0
        self._wall = time.time

    # ── setup ───────────────────────────────────────────────────────────
    def load_model(self) -> None:
        import cv2
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
        self._cv2, self._mp = cv2, mp
        if not os.path.isfile(self.args.model):
            raise FileNotFoundError(f"hand model not found: {self.args.model}")
        opts = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.args.model),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(opts)
        log(f"model loaded ({os.path.basename(self.args.model)}), "
            f"mediapipe {mp.__version__}, opencv {cv2.__version__}")

    def _attach(self) -> None:
        now = time.monotonic()
        if self.shm is not None or now - self._last_attach_try < 0.5:
            return
        self._last_attach_try = now
        try:
            from multiprocessing import shared_memory
            self.shm = shared_memory.SharedMemory(name=self.args.shm_name)
            log(f"attached to shared memory {self.args.shm_name} "
                f"({self.shm.size} bytes)")
        except FileNotFoundError:
            self.shm = None
        except Exception as e:
            log(f"shared memory attach failed: {type(e).__name__}: {e}")
            self.shm = None

    def _attach_depth(self) -> None:
        now = time.monotonic()
        if (self.depthmod is None or self.depth_shm is not None
                or now - self._last_depth_attach_try < 1.0):
            return
        self._last_depth_attach_try = now
        try:
            from multiprocessing import shared_memory
            self.depth_shm = shared_memory.SharedMemory(name=self.args.depth_shm_name)
            log(f"attached to depth segment {self.args.depth_shm_name}")
        except FileNotFoundError:
            self.depth_shm = None
        except Exception as e:
            log(f"depth segment attach failed: {type(e).__name__}: {e}")
            self.depth_shm = None

    def _poll_depth(self) -> None:
        """Take a NEW depth plane from the segment (if any), mirror it, relay it
        to the clients that subscribed to depth. Never raises."""
        try:
            self._attach_depth()
            if self.depth_shm is None:
                return
            got = self.depthmod.read_depth(self.depth_shm.buf, self._depth_seq)
            if got is None:
                return
            import numpy as np
            seq, color_seq, ts_ms, plane = got
            self._depth_seq = seq
            self._depth_plane = np.ascontiguousarray(plane[:, ::-1])
            self._depth_color_seq = color_seq
            self._depth_ts_ms = ts_ms
            if self.hub.wants_depth():
                self.hub.broadcast(depth=self.depthmod.pack_wire(
                    self._depth_plane, color_seq, mirrored=True))
                self._depth_relayed += 1
        except Exception as e:
            log(f"depth poll failed: {type(e).__name__}: {e}")

    def depth_fresh(self) -> bool:
        """True while the newest plane is younger than the module's
        FRESH_DEPTH_S (wall clock: JARVIS stamps it, same machine)."""
        if self._depth_plane is None or self.depthmod is None:
            return False
        age_ms = self._wall() * 1000.0 - self._depth_ts_ms
        return age_ms <= self.depthmod.FRESH_DEPTH_S * 1000.0

    def _hand_z(self, palm) -> Optional[float]:
        """Palm depth in metres: median of a 5x5 window of the plane, or None."""
        try:
            return self.depthmod.sample_median_m(self._depth_plane,
                                                 float(palm[0]), float(palm[1]))
        except Exception:
            return None

    # ── lifecycle ───────────────────────────────────────────────────────
    def _beat(self, now: float) -> None:
        if now - self._last_beat < 1.0 or not self.args.alive:
            return
        self._last_beat = now
        status = {"pid": os.getpid(), "t": round(time.time(), 2),
                  "fps": round(self.fps, 1), "frames": self.processed,
                  "hands": self.hands_now, "clients": self.hub.client_count(),
                  "attached": self.shm is not None, "port": self.args.port,
                  "depth": self.depth_fresh(),
                  "up_s": round(time.time() - self.started, 1)}
        try:
            os.makedirs(os.path.dirname(self.args.alive) or ".", exist_ok=True)
            with open(self.args.alive, "w", encoding="utf-8") as f:
                json.dump(status, f)
        except OSError as e:
            log(f"heartbeat write failed: {e}")

    def _jarvis_gone(self, now: float) -> Optional[str]:
        if now - self._last_exit_check < 0.5:
            return None
        self._last_exit_check = now
        a = self.args
        if a.jarvis_pid and not pid_alive(a.jarvis_pid):
            return f"JARVIS (pid {a.jarvis_pid}) is gone"
        if a.jarvis_alive:
            age = file_age(a.jarvis_alive)
            if age is None:
                if self._jarvis_seen:
                    return "JARVIS removed its heartbeat (hand tracking off)"
                if time.time() - self.started > a.grace_s:
                    return "JARVIS heartbeat never appeared"
            elif age > a.stale_s:
                return f"JARVIS heartbeat stale ({age:.0f} s)"
            else:
                self._jarvis_seen = True
        return None

    # ── the loop ────────────────────────────────────────────────────────
    def run(self) -> int:
        a = self.args
        if a.host not in LOOPBACK:
            log(f"refusing to bind {a.host}: loopback only")
            return 2
        try:
            self.load_model()
        except Exception as e:
            log(f"cannot start: {type(e).__name__}: {e}")
            return 3
        if not self.hub.start():
            log("cannot start the websocket server")
            self._cleanup()
            return 4
        for sig in ("SIGINT", "SIGTERM", "SIGBREAK"):
            s = getattr(signal, sig, None)
            if s is not None:
                try:
                    signal.signal(s, lambda *_: self.stop.set())
                except Exception:
                    pass
        log("ready; waiting for frames from JARVIS")
        last_seq = 0
        reason = "stopped"
        try:
            while not self.stop.is_set():
                now = time.monotonic()
                self._beat(now)
                gone = self._jarvis_gone(now)
                if gone:
                    reason = gone
                    break
                self._attach()
                self._poll_depth()
                got = None
                if self.shm is not None:
                    got = self.frames.read_frame(self.shm.buf, last_seq)
                if got is None:
                    self._idle(now)
                    time.sleep(0.004)
                    continue
                seq, ts_ms, frame = got
                last_seq = seq
                self._last_frame_at = now
                self._process(seq, ts_ms, frame)
        except Exception as e:
            reason = f"crashed: {type(e).__name__}: {e}"
            import traceback
            traceback.print_exc()
        log(f"exiting: {reason}")
        self._cleanup()
        return 0

    def _idle(self, now: float) -> None:
        """No new frame: after 0.5 s tell clients the hands are gone (once),
        so a held fist releases and the page drops its cursors."""
        if (not self._empty_sent and self._last_frame_at
                and now - self._last_frame_at > 0.5):
            self._empty_sent = True
            W, H = self._last_wh
            _infos, events = self.engine.update(now, [], W, H)
            self.hands_now = 0
            self.hub.broadcast(json.dumps(
                hands_message(int(time.time() * 1000), 0, W, H, []),
                separators=(",", ":")))
            for ev in events:
                self.hub.broadcast(json.dumps(ev, separators=(",", ":")))

    def _process(self, seq: int, ts_ms: int, frame) -> None:
        cv2, mp = self._cv2, self._mp
        H, W = frame.shape[:2]                  # the frame's real size (aspect kept)
        self._last_wh = (W, H)
        mirrored = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(mirrored, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ms = max(self._last_ts_ms + 1, int(time.monotonic() * 1000))
        self._last_ts_ms = ms
        result = self.landmarker.detect_for_video(image, ms)
        raw = []
        for i, lms in enumerate(result.hand_landmarks or []):
            cat = result.handedness[i][0] if i < len(result.handedness) else None
            side = (cat.category_name if cat else "Right")
            if side not in ("Left", "Right"):
                side = "Right"
            world = (result.hand_world_landmarks[i]
                     if i < len(result.hand_world_landmarks) else [])
            raw.append({
                "side": side, "score": float(cat.score) if cat else 0.0,
                "lm": [[p.x, p.y, p.z] for p in lms],
                "world": [[p.x, p.y, p.z] for p in world] or [[0.0, 0.0, 0.0]] * 21,
            })
        now = time.monotonic()
        infos, events = self.engine.update(now, raw, W, H)
        payload = [hand_payload(info, hd["lm"], hd["world"])
                   for info, hd in zip(infos, raw) if info["id"] >= 0]
        has_depth = self.depth_fresh()
        if has_depth:
            for hp in payload:
                hp["z"] = self._hand_z(hp["palm"])
        self.hands_now = len(payload)
        self._empty_sent = not payload
        self.hub.broadcast(json.dumps(
            hands_message(ts_ms, seq, W, H, payload, depth=has_depth),
            separators=(",", ":")))
        for ev in events:
            log(f"gesture {ev['name']} {ev['side']} {ev['pos']}"
                + (f" {ev['value']}" if 'value' in ev else ""))
            self.hub.broadcast(json.dumps(ev, separators=(",", ":")))
        if self.hub.wants_frames():
            ok, buf = cv2.imencode(".jpg", mirrored,
                                   [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if ok:
                self.hub.broadcast(jpeg=buf.tobytes())
        self.processed += 1
        self._fps_n += 1
        if now - self._fps_t0 >= 2.0:
            self.fps = self._fps_n / (now - self._fps_t0)
            self._fps_t0, self._fps_n = now, 0

    def _cleanup(self) -> None:
        try:
            self.hub.stop()
        except Exception:
            pass
        if self.landmarker is not None:
            try:
                self.landmarker.close()
            except Exception:
                pass
        for seg in (self.shm, self.depth_shm):
            if seg is not None:
                try:
                    seg.close()
                except Exception:
                    pass
        if self.args.alive:
            try:
                os.remove(self.args.alive)
            except OSError:
                pass


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="JARVIS hand tracker service")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--shm-name", default="jarvis_handtrack_frame")
    p.add_argument("--depth-shm-name", default="jarvis_handtrack_depth",
                   help="Kinect depth plane segment (optional; absent = no depth)")
    p.add_argument("--alive", default=os.path.join(ROOT, "data", ".handtrack_alive"),
                   help="heartbeat file this service refreshes (JARVIS's frame "
                        "publisher runs only while it is fresh)")
    p.add_argument("--jarvis-pid", type=int, default=0,
                   help="exit when this process is gone")
    p.add_argument("--jarvis-alive", default="",
                   help="file JARVIS touches; exit when it goes stale")
    p.add_argument("--stale-s", type=float, default=15.0)
    p.add_argument("--grace-s", type=float, default=30.0)
    p.add_argument("--invert-palm", action="store_true",
                   help="swap palm/back-of-hand if the camera reports it reversed")
    return p.parse_args(argv)


def main(argv=None) -> int:
    return Tracker(parse_args(argv)).run()


if __name__ == "__main__":
    sys.exit(main())
