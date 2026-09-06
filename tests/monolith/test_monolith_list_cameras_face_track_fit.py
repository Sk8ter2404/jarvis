"""``--list-cameras`` must not recommend an index the face tracker cannot use.

────────────────────────────────────────────────────────────────────────────
WHAT WENT WRONG
────────────────────────────────────────────────────────────────────────────
list_cameras() exists to answer exactly one question for the owner: "which
index do I paste into CAMERAS?" It sweeps at 1920x1080 and prints
``Camera N: working (WxH)``.

A CAMERAS entry, though, is opened by the face-track producer at
``_LOOP_OPEN_WIDTH x _LOOP_OPEN_HEIGHT`` (1280x720) WITH a require_frame
budget. Those are not the same question, and on the owner's rig they have
different answers. Measured 2026-09-06, JARVIS stopped, through the shipped
opener ``core.camera_backend.open_camera``, a fresh process per figure:

    Kinect V2, MSMF idx 0, 1920x1080 : 30.05-30.10 fps, 91/91 reads in 3 s,
                                       first frame 0.03 s, brightness ~122 (n=3)
    Kinect V2, MSMF idx 0, 1280x720  : get() reads back 512x424 and EVERY read
                                       fails — 0 frames in 5 s. 640x480 and
                                       "no resolution set" are identical (n=2 ea)
    Kinect V2, DSHOW idx 1, 1920x1080: 1.0 fps and mean brightness 0.0 (black)
                                       in 5 of 7 tries, 15.1 fps once, 1 refused
                                       open. Requested size ignored entirely.

So the real ``--list-cameras`` run on 2026-09-06 printed
``Camera 1: working (1920x1080) [OK, brightness=124.4]`` — the BRIGHTEST entry
in the sweep — for a device the face tracker then drops with "Could not open …
skipping". The tool's own closing line ("Then edit the CAMERAS list") is what
turns that into a wrong action by the owner.

(The docstring these tests replace claimed the opposite: that Media Foundation
"never delivers a frame" from this device, was "strictly WORSE" for it, and
that the Kinect therefore "drops out of the listing". All three are falsified
by the numbers above; the sensor is listed, and at the sweep's own resolution
MSMF is ~30x better here.)

────────────────────────────────────────────────────────────────────────────
WHAT IS ASSERTED
────────────────────────────────────────────────────────────────────────────
Behaviour, not this rig. No test here needs a camera, a backend, or a device
list: a fake opener stands in for "a device with one fixed format". What is
pinned is that the sweep asks the SECOND question at the producer's own
resolution constant, believes a "no" only after a retry, and says so loudly —
and that it still merely PRINTS, rewriting nothing.
"""
from __future__ import annotations

import unittest
from unittest import mock

from tests._monolith_harness import MonolithGlobalsTestCase, requires_monolith


def _executable_source(fn) -> str:
    """``fn``'s source with comments AND string literals removed.

    Docstrings go too, deliberately: these functions describe measured device
    behaviour in prose, and a substring search over raw source cannot tell a
    description from a call. Whitespace is stripped so ``cv2.VideoCapture(``
    cannot hide as ``cv2 . VideoCapture (``."""
    import inspect
    import io
    import tokenize
    src = inspect.getsource(fn)
    skip = {tokenize.COMMENT, tokenize.STRING, tokenize.NL, tokenize.NEWLINE,
            tokenize.INDENT, tokenize.DEDENT, tokenize.FSTRING_START,
            tokenize.FSTRING_MIDDLE, tokenize.FSTRING_END}
    out = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type in skip:
                continue
            out.append(tok.string.strip())
    except tokenize.TokenError:      # pragma: no cover - defensive
        return src
    return "".join(out)


class _Cap:
    """A capture handle. ``frames=False`` is the shape that has cost this
    project evenings: isOpened() was True, set() was accepted, and no image
    ever arrives."""

    def __init__(self, frames: bool = True):
        self._frames = frames
        self.released = 0

    def read(self):
        if not self._frames:
            return False, None
        import numpy as np
        return True, np.full((16, 16, 3), 40, dtype="uint8")

    def get(self, *_a):
        return 0.0

    def set(self, *_a):
        return True

    def release(self):
        self.released += 1


@requires_monolith
class ListCamerasFaceTrackFitTests(MonolithGlobalsTestCase):
    """The confirm probe, and the advice it produces."""

    def _run_sweep(self, opener, max_check=1):
        """Drive list_cameras() with ``opener`` standing in for _camera_open.

        Returns (printed_lines, opener_calls)."""
        bc = self.bc
        printed: list = []
        calls: list = []

        def _spy(idx, *, name=None, width=None, height=None,
                 require_frame=0.0, label=""):
            calls.append({"idx": idx, "width": width, "height": height,
                          "label": label})
            return opener(idx, width, height)

        with mock.patch.object(bc, "_camera_open", side_effect=_spy), \
                mock.patch.object(bc, "find_camera_locking_processes",
                                  return_value=[]), \
                mock.patch.object(bc.cv2, "imwrite", return_value=True), \
                mock.patch("builtins.print", printed.append):
            bc.list_cameras(max_check=max_check)
        return [str(line) for line in printed], calls

    # ── the device with ONE fixed format ────────────────────────────────
    @staticmethod
    def _one_fixed_format(only_w=1920, only_h=1080):
        """The Kinect's measured shape: frames at exactly one size, and a
        handle that opens and delivers nothing at every other size."""
        def _open(idx, width, height):
            return _Cap(frames=(width == only_w and height == only_h))
        return _open

    def test_a_one_format_camera_is_listed_working_and_flagged_do_not_use(self):
        """It IS listed — the old docstring's "drops out of the listing" was
        wrong — but it is not silently recommended."""
        lines, _ = self._run_sweep(self._one_fixed_format())
        blob = "\n".join(lines)
        self.assertIn("Camera 0: working", blob,
                      "a device that delivers at the sweep resolution must "
                      "still be listed as working")
        self.assertIn("DO NOT PUT INDEX 0 IN CAMERAS", blob,
                      "the sweep recommended an index the face-track producer "
                      "cannot open — the exact wrong action this tool invites")

    def test_the_confirm_probe_asks_at_the_producers_own_resolution(self):
        bc = self.bc
        _, calls = self._run_sweep(self._one_fixed_format())
        confirm = [c for c in calls
                   if (c["width"], c["height"]) == (bc._LOOP_OPEN_WIDTH,
                                                    bc._LOOP_OPEN_HEIGHT)]
        self.assertTrue(confirm,
                        "no probe was made at the resolution a CAMERAS entry "
                        "is actually opened at")
        # Sweep resolution first, producer resolution second — asking the
        # second question of an index that failed the first would waste an
        # open on a dead index, which is what costs handles on DirectShow.
        self.assertEqual((calls[0]["width"], calls[0]["height"]), (1920, 1080))

    def test_an_ordinary_webcam_is_not_flagged(self):
        """A camera that works at both sizes must draw no warning at all — a
        false "DO NOT PUT INDEX 0 IN CAMERAS" about the owner's own webcam is
        worse advice than the silence it replaced."""
        lines, calls = self._run_sweep(lambda idx, w, h: _Cap(frames=True))
        blob = "\n".join(lines)
        self.assertIn("Camera 0: working", blob)
        self.assertNotIn("DO NOT PUT", blob)
        self.assertNotIn("Skip", blob)
        # And it cost exactly one confirm open, not the retry pair.
        self.assertEqual(len(calls), 2, calls)

    def test_a_transient_frameless_confirm_is_retried_before_it_is_believed(self):
        """MSMF fails 3-7% of back-to-back reopens (measured; see
        core.camera_backend.default_retries), and a camera another app grabs
        between the two probes looks identical to a format refusal. One flake
        must not produce do-not-use advice about a working webcam."""
        state = {"confirms": 0}

        def _open(idx, width, height):
            if (width, height) == (1920, 1080):
                return _Cap(frames=True)
            state["confirms"] += 1
            return _Cap(frames=state["confirms"] > 1)   # first confirm flakes

        lines, _ = self._run_sweep(_open)
        self.assertEqual(state["confirms"], 2)
        self.assertNotIn("DO NOT PUT", "\n".join(lines))

    def test_two_frameless_confirms_in_a_row_are_believed(self):
        lines, _ = self._run_sweep(self._one_fixed_format())
        self.assertIn("DO NOT PUT", "\n".join(lines))

    def test_the_closing_summary_names_every_flagged_index(self):
        lines, _ = self._run_sweep(self._one_fixed_format(), max_check=2)
        blob = "\n".join(lines)
        self.assertIn("Skip 0, 1", blob,
                      "the flagged indices must be repeated next to the "
                      "'edit the CAMERAS list' instruction, which is the line "
                      "the owner actually acts on")

    # ── the other two claims the old docstring got wrong ────────────────
    def test_a_frameless_index_is_reported_not_dropped(self):
        """The old docstring said a frameless device "drops out of the
        listing". It never did: _open_with_warmup marks any non-None capture
        opened, so the sweep prints "opened but no frame"."""
        lines, calls = self._run_sweep(lambda idx, w, h: _Cap(frames=False))
        blob = "\n".join(lines)
        self.assertIn("Camera 0: opened but no frame", blob)
        self.assertNotIn("not available", blob)
        # An index that never delivered is not something we were about to
        # recommend, so it buys no second open.
        self.assertEqual(len(calls), 1, calls)

    def test_a_dead_index_is_reported_not_available_and_buys_no_confirm(self):
        lines, calls = self._run_sweep(lambda idx, w, h: None)
        self.assertIn("Camera 0: not available", "\n".join(lines))
        self.assertEqual(len(calls), 1, calls)

    def test_the_sweep_rewrites_nothing(self):
        """The old docstring justified excluding the Kinect with "this sweep
        REWRITES CAMERAS". It does not — it prints, and tells the owner to
        edit the file. The function that rewrites CAMERAS is
        probe_cameras_and_update_config(). Keep them separate: a sweep that
        started writing would turn every warning above into a silent
        misconfiguration."""
        bc = self.bc
        code = _executable_source(bc.list_cameras)
        for forbidden in ("save_user_settings", "_write_user_settings",
                          "probe_cameras_and_update_config", "CAMERAS="):
            self.assertNotIn(forbidden, code,
                             f"list_cameras() now reaches {forbidden!r} — it "
                             f"is a read-only diagnostic")

    def test_the_producer_and_the_sweep_share_one_resolution_constant(self):
        """A second hardcoded 1280x720 is the stale-duplicate bug shape this
        codebase pays for most: the producer's resolution could move and the
        sweep would keep certifying indices against the old one."""
        bc = self.bc
        producer = _executable_source(bc._face_tracking_thread_body)
        sweep = _executable_source(bc.list_cameras)
        for name in ("_LOOP_OPEN_WIDTH", "_LOOP_OPEN_HEIGHT"):
            self.assertIn(name, producer,
                          f"the face-track producer no longer uses {name}")
            self.assertIn(name, sweep,
                          f"list_cameras() no longer uses {name}")
        self.assertEqual((bc._LOOP_OPEN_WIDTH, bc._LOOP_OPEN_HEIGHT),
                         (1280, 720))


if __name__ == "__main__":
    unittest.main()
