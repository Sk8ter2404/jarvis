"""Headless tests for the Bambu chamber-camera feature.

Covers BOTH halves of the feature without a real printer, a display, or the
optional GUI/vision deps the CI runner doesn't install:

  • core/bambu_camera.py — the LAN frame grabber. Loaded with ``cv2`` forced
    ABSENT (exactly as on the Linux CI runner, where cv2 is in the block
    list), proving the module imports and its stdlib JPEG-stills path works
    with no OpenCV. The protocol logic (80-byte auth packet, JPEG SOI..EOI
    framing, RTSPS URL, model→path classification, cv2-aware path ordering)
    is asserted directly, and the full stills grab is driven end-to-end
    against a FAKE ssl socket so the read-loop + atomic frame/state writes
    are exercised offline.

  • hud/bambu_camera_hud.py — the PyQt6 view. Loaded with PyQt6 forced
    ABSENT (its ``except ImportError`` stub path), so its non-Qt helpers
    (frame-age, offline-reason, footer formatting) are unit-tested on the
    headless runner with no Qt loop ever spun. (The dedicated
    tests/test_hud_pyqt6_absent.py separately proves the module *imports*
    clean without PyQt6; here we exercise its logic.)

ISOLATION
  • cv2 / PyQt6 imports are blocked for the duration of the relevant loads
    (a fake ``__import__`` raises ImportError for them), and any previously
    cached submodules are hidden during the load and restored after — so a
    dev box that HAS them still exercises the headless path.
  • FRAME_FILE / STATE_FILE / config files are redirected into a per-test
    temp dir; no real project file is read or written, and no real network
    socket is opened (the grab path is driven through an injected fake).
  • ``bobert_companion`` is never booted — config reads are pinned via a
    fake module in sys.modules (restored, including absence, on cleanup).

stdlib ``unittest`` + ``unittest.mock`` only (no pytest); App-Control-safe.
"""
from __future__ import annotations

import contextlib
import importlib
import importlib.util
import os
import struct
import sys
import tempfile
import types
import unittest
from unittest import mock


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

_SENTINEL = object()


@contextlib.contextmanager
def _block_imports(*blocked_heads):
    """Block ``import <head>...`` for the with-block (raising ImportError, as
    on the CI runner) and hide any already-loaded copies, restoring everything
    — including the real importer and cached modules — on exit."""
    blocked = set(blocked_heads)
    real_import = __import__

    def _imp(name, *a, **k):
        if name.split(".")[0] in blocked:
            raise ImportError(f"[test] blocked: {name}")
        return real_import(name, *a, **k)

    hidden = {n: sys.modules.pop(n) for n in list(sys.modules)
              if n.split(".")[0] in blocked}
    with mock.patch("builtins.__import__", _imp):
        try:
            yield
        finally:
            for n in list(sys.modules):
                if n.split(".")[0] in blocked:
                    del sys.modules[n]
            sys.modules.update(hidden)


@contextlib.contextmanager
def _fake_bobert(**flags):
    """Pin a fake ``bobert_companion`` in sys.modules so the grabber's lazy
    config reads resolve to ``flags`` without booting the monolith. Restores
    the prior entry (including absence) on exit."""
    prev = sys.modules.get("bobert_companion", _SENTINEL)
    bc = types.ModuleType("bobert_companion")
    for k, v in flags.items():
        setattr(bc, k, v)
    sys.modules["bobert_companion"] = bc
    try:
        yield bc
    finally:
        if prev is _SENTINEL:
            sys.modules.pop("bobert_companion", None)
        else:
            sys.modules["bobert_companion"] = prev


def _load_bambu_camera_no_cv2():
    """Import core/bambu_camera.py with cv2 blocked, under a fresh module name
    so module globals (status dict, file paths) start clean. Returns the
    module; caller drops it from sys.modules on cleanup."""
    path = os.path.join(_PROJECT_ROOT, "core", "bambu_camera.py")
    with _block_imports("cv2"):
        spec = importlib.util.spec_from_file_location("_test_bambu_camera", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["_test_bambu_camera"] = mod
        spec.loader.exec_module(mod)
    return mod


def _load_camera_hud_no_pyqt():
    """Load hud/bambu_camera_hud.py with PyQt6 blocked (graceful-degrade path)
    under a synthetic module name. Returns the module."""
    path = os.path.join(_PROJECT_ROOT, "hud", "bambu_camera_hud.py")
    with _block_imports("PyQt6"):
        spec = importlib.util.spec_from_file_location(
            "_test_bambu_camera_hud", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["_test_bambu_camera_hud"] = mod
        spec.loader.exec_module(mod)
    return mod


class FakeSslSock:
    """A stand-in for the TLS socket the JPEG-stills grabber talks to.

    ``recv`` dispenses the queued chunks (then b"" to signal stream end);
    ``sendall`` records the auth packet so the test can assert it. No real
    network is touched."""

    def __init__(self, chunks):
        self._chunks = list(chunks)
        self.sent = b""
        self.closed = False

    def settimeout(self, _t):
        pass

    def sendall(self, data):
        self.sent += data

    def recv(self, _n):
        if self._chunks:
            return self._chunks.pop(0)
        return b""

    def close(self):
        self.closed = True


# ─────────────────────────────────────────────────────────────────────────
#  core/bambu_camera.py — grabber
# ─────────────────────────────────────────────────────────────────────────
class BambuCameraGrabberTests(unittest.TestCase):
    def setUp(self):
        self.mod = _load_bambu_camera_no_cv2()
        # Redirect all artifact paths into a throwaway dir.
        self.tmp = tempfile.mkdtemp(prefix="bambucam_test_")
        self.mod._DATA_DIR = self.tmp
        self.mod.FRAME_FILE = os.path.join(self.tmp, "frame.jpg")
        self.mod.STATE_FILE = os.path.join(self.tmp, "state.json")

    def tearDown(self):
        try:
            self.mod.stop_grabber()
        except Exception:
            pass
        sys.modules.pop("_test_bambu_camera", None)
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    # ── import contract ──
    def test_imports_without_cv2(self):
        self.assertFalse(self.mod._HAS_CV2,
                         "bambu_camera must report _HAS_CV2 False on CI")

    # ── auth packet ──
    def test_auth_packet_layout(self):
        pkt = self.mod.build_auth_packet("bblp", "12345678")
        self.assertEqual(len(pkt), 80)
        self.assertEqual(struct.unpack("<IIII", pkt[:16]), (0x40, 0x3000, 0, 0))
        self.assertEqual(pkt[16:20], b"bblp")
        self.assertEqual(pkt[20:48], b"\x00" * 28)        # username padding
        self.assertEqual(pkt[48:56], b"12345678")
        self.assertEqual(pkt[56:80], b"\x00" * 24)        # access-code padding

    def test_auth_packet_handles_long_code(self):
        # An over-length code is truncated to 32 bytes, not crash.
        pkt = self.mod.build_auth_packet("bblp", "x" * 40)
        self.assertEqual(len(pkt), 80)
        self.assertEqual(pkt[48:80], b"x" * 32)

    # ── JPEG framing ──
    def test_extract_jpeg_full_frame_with_junk(self):
        soi, eoi = b"\xff\xd8\xff\xe0", b"\xff\xd9"
        body = soi + b"PIXELS" + eoi
        frame, rest = self.mod.extract_jpeg(b"\x00junk" + body + b"NEXT")
        self.assertEqual(frame, body)
        self.assertEqual(rest, b"NEXT")

    def test_extract_jpeg_partial_returns_none(self):
        soi = b"\xff\xd8\xff\xe0"
        frame, rest = self.mod.extract_jpeg(b"z" + soi + b"half")
        self.assertIsNone(frame)
        self.assertEqual(rest, soi + b"half")        # buffer kept from SOI

    def test_extract_jpeg_no_soi_trims(self):
        frame, rest = self.mod.extract_jpeg(b"no markers here")
        self.assertIsNone(frame)
        self.assertLessEqual(len(rest), 3)           # only a small tail kept

    # ── URL + model classification ──
    def test_rtsps_url(self):
        self.assertEqual(
            self.mod.build_rtsps_url("192.0.2.5", "CODE"),
            "rtsps://bblp:CODE@192.0.2.5:322/streaming/live/1",
        )

    def test_model_classification(self):
        c = self.mod.camera_supported_for_model
        self.assertEqual(c("H2D"), "rtsps")
        self.assertEqual(c("X1C"), "rtsps")
        self.assertEqual(c("P1S"), "jpeg")
        self.assertEqual(c("A1 mini"), "jpeg")
        self.assertEqual(c(""), "unknown")
        self.assertEqual(c("ZZZ9"), "unknown")

    def test_path_order_drops_rtsps_without_cv2(self):
        # cv2 is absent in this module, so RTSPS (which needs it) must be
        # dropped from the order regardless of model.
        self.assertEqual(self.mod._ordered_paths("H2D"), ["jpeg"])
        self.assertEqual(self.mod._ordered_paths("P1S"), ["jpeg"])
        self.assertEqual(self.mod._ordered_paths(""), ["jpeg"])

    # ── full stills grab against a fake socket ──
    def test_grab_once_jpeg_path_writes_frame_and_state(self):
        soi, eoi = b"\xff\xd8\xff\xe0", b"\xff\xd9"
        jpeg = soi + b"\x01\x02\x03IMAGE" + eoi
        # Split the frame across two recv() chunks to exercise reassembly.
        fake = FakeSslSock([b"\x00pre" + jpeg[:6], jpeg[6:] + b"tail"])
        fake_ctx = mock.MagicMock()
        fake_ctx.wrap_socket.return_value = fake
        with mock.patch.object(self.mod.ssl, "create_default_context",
                               return_value=fake_ctx), \
             mock.patch.object(self.mod.socket, "create_connection",
                               return_value=mock.MagicMock()):
            out = self.mod.grab_once("192.0.2.9", "12345678", model_hint="P1S")
        self.assertEqual(out, jpeg, "grab_once should return the JPEG frame")
        # Auth packet was sent first.
        self.assertEqual(fake.sent, self.mod.build_auth_packet("bblp", "12345678"))
        # Frame written to disk.
        with open(self.mod.FRAME_FILE, "rb") as f:
            self.assertEqual(f.read(), jpeg)
        # Status sidecar reflects success via the JPEG path.
        st = self.mod.get_status()
        self.assertTrue(st["ok"])
        self.assertEqual(st["path"], "jpeg")
        self.assertGreaterEqual(st["frame_count"], 1)

    def test_grab_once_no_frame_records_error(self):
        # A socket that only yields non-JPEG bytes then EOF -> no frame.
        fake = FakeSslSock([b"not a jpeg at all"])
        fake_ctx = mock.MagicMock()
        fake_ctx.wrap_socket.return_value = fake
        with mock.patch.object(self.mod.ssl, "create_default_context",
                               return_value=fake_ctx), \
             mock.patch.object(self.mod.socket, "create_connection",
                               return_value=mock.MagicMock()):
            out = self.mod.grab_once("192.0.2.9", "12345678", model_hint="P1S")
        self.assertIsNone(out)
        st = self.mod.get_status()
        self.assertFalse(st["ok"])
        self.assertIsNotNone(st["last_error"])

    def test_grab_once_unconfigured_is_safe(self):
        out = self.mod.grab_once("", "", model_hint="H2D")
        self.assertIsNone(out)
        self.assertIn("credential", (self.mod.get_status()["last_error"] or ""))

    def test_jpeg_socket_connection_failure_returns_none(self):
        # create_connection raising (printer asleep) must be swallowed.
        with mock.patch.object(self.mod.socket, "create_connection",
                               side_effect=OSError("refused")):
            out = self.mod._grab_jpeg_once("192.0.2.9", "code", timeout=0.1)
        self.assertIsNone(out)

    # ── config gating + lifecycle ──
    def test_start_disabled_by_flag(self):
        with _fake_bobert(HUD_BAMBU_CAMERA=False,
                          BAMBU_PRINTER_IP="192.0.2.1",
                          BAMBU_ACCESS_CODE="12345678",
                          BAMBU_SERIAL="0309ABC"):
            self.assertFalse(self.mod.start_grabber())
            self.assertFalse(self.mod.is_running())
        st = self.mod.get_status()
        self.assertIn("disabled", (st["last_error"] or "").lower())

    def test_start_unconfigured_no_thread(self):
        with _fake_bobert(HUD_BAMBU_CAMERA=True,
                          BAMBU_PRINTER_IP="", BAMBU_ACCESS_CODE="",
                          BAMBU_SERIAL=""):
            self.assertFalse(self.mod.start_grabber())
            self.assertFalse(self.mod.is_running())

    def test_start_is_idempotent_and_stoppable(self):
        # Patch the poll loop to a no-op so no real network/cv2 is touched;
        # we're only asserting the thread lifecycle + idempotency here.
        with _fake_bobert(HUD_BAMBU_CAMERA=True,
                          BAMBU_PRINTER_IP="192.0.2.1",
                          BAMBU_ACCESS_CODE="12345678",
                          BAMBU_SERIAL="0309ABC"), \
             mock.patch.object(self.mod, "_poll_loop",
                               side_effect=lambda evt: evt.wait()):
            self.assertTrue(self.mod.start_grabber())
            self.assertTrue(self.mod.is_running())
            t1 = self.mod._grab_thread[0]
            # Second call must NOT spawn a second thread.
            self.assertTrue(self.mod.start_grabber())
            self.assertIs(self.mod._grab_thread[0], t1)
            self.mod.stop_grabber()
            self.assertFalse(self.mod.is_running())


# ─────────────────────────────────────────────────────────────────────────
#  hud/bambu_camera_hud.py — view helpers (PyQt6 absent)
# ─────────────────────────────────────────────────────────────────────────
class BambuCameraHudHelpersTests(unittest.TestCase):
    def setUp(self):
        self.hud = _load_camera_hud_no_pyqt()

    def tearDown(self):
        sys.modules.pop("_test_bambu_camera_hud", None)

    def test_imports_without_pyqt6(self):
        self.assertFalse(self.hud._HAS_PYQT6,
                         "camera HUD must report _HAS_PYQT6 False on CI")

    def test_main_returns_2_without_pyqt6(self):
        with mock.patch.object(sys, "argv", ["x", "--parent-pid", "0"]):
            self.assertEqual(self.hud.main(), 2)

    def test_format_minutes(self):
        f = self.hud._format_minutes
        self.assertEqual(f(0), "<1m")
        self.assertEqual(f(5), "5m")
        self.assertEqual(f(60), "1h")
        self.assertEqual(f(108), "1h 48m")
        self.assertEqual(f(None), "")

    def test_shorten(self):
        self.assertEqual(self.hud._shorten("short.3mf", 30), "short.3mf")
        self.assertTrue(self.hud._shorten("x" * 50, 10).endswith("…"))
        self.assertEqual(self.hud._shorten("", 30), "")

    def test_frame_age_prefers_timestamp(self):
        import time as _t
        age = self.hud._frame_age({"last_frame_at": _t.time() - 5})
        self.assertGreaterEqual(age, 4.0)
        self.assertLess(age, 8.0)

    def test_frame_age_missing_is_huge(self):
        # No timestamp and (in this temp context) no frame file -> huge age,
        # which the widget reads as "stale -> show placeholder".
        self.hud.FRAME_FILE = os.path.join(tempfile.gettempdir(),
                                           "definitely_missing_frame.jpg")
        self.assertGreater(self.hud._frame_age({}), 1000.0)

    def test_offline_reason_maps_errors(self):
        r = self.hud._offline_reason
        self.assertIn("credential", r({"last_error": "credentials not configured"}, {}).lower())
        self.assertIn("disabled", r({"last_error": "HUD_BAMBU_CAMERA disabled"}, {}).lower())
        self.assertIn("LAN Only Liveview", r({"last_error": "jpeg path returned no frame"}, {}))
        # No grabber error -> fall back to printer state.
        self.assertIn("idle", r({}, {"gcode_state": "IDLE"}).lower())
        self.assertIn("offline", r({}, {}).lower())


if __name__ == "__main__":
    unittest.main()
