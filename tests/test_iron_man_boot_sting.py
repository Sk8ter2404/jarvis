"""The boot sting's PortAudio stream must be owned, and must be closed.

2026-08-20 adversarial review, MEDIUM: ``iron_man_boot._play_sting_async``
called ``sd.play(audio, ...)`` — which opens a live OutputStream with a running
PortAudio callback and publishes it into sounddevice's module-global
``_last_callback`` — while claiming NOTHING. For the whole ~1.5 s sting every
owner cell bobert's teardown gate reads was clear, and the stream was never
closed: ``_play_sting_async`` returned immediately, no reaper was attached, and
the handle lived until the next ``sd.play()``'s internal ``stop()`` happened to
reap it.

Boot is precisely when the wake listener, the ambient workers, the diagnostic
daemons and the face tracker all start calling
``get_input_device()``/``get_output_device()``. The first one to do so runs
``_refresh_devices`` against a first-pass (None) baseline, sees drift, finds no
owner flag and no pending close, and runs ``sd._terminate()``/``sd._initialize()``
with the sting's callback thread live — the 0xc0000374 heap corruption the gate
exists to prevent, at the worst possible moment (no traceback; the watchdog just
sees a vanished process). The BUFFER-lifetime half of this hazard was fixed
earlier (``_LAST_STING_BUF`` + the full-duration sleep); this is the
PortAudio-teardown half.

The module is deliberately self-contained — it imports nothing from
bobert_companion — so every assertion here also pins the DEGRADED path: with no
monolith loaded (a stand-alone run) the sting must still play.
"""
from __future__ import annotations

import os
import sys
import types
import unittest
from unittest import mock

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import iron_man_boot as ib  # noqa: E402


class _FakeSd(types.ModuleType):
    def __init__(self, stream=None, play_raises=False):
        super().__init__("sounddevice")
        self.played = []
        self._stream = stream
        self._play_raises = play_raises

        def _play(audio, **kwargs):
            if self._play_raises:
                raise RuntimeError("no output device")
            self.played.append((audio, kwargs))

        def _get_stream():
            if self._stream is None:
                raise RuntimeError("no stream")
            return self._stream

        self.play = _play
        self.get_stream = _get_stream


def _fake_bc(*, claim_ok=True, with_closer=True):
    """A monolith stand-in exposing exactly the gate surface the sting uses."""
    state = {"claims": [], "releases": [], "closed": [], "detached": []}
    cell = [False]

    def _claim(c, *, refcount=False, timeout=1.0, deny_if=None):
        state["claims"].append(timeout)
        if not claim_ok:
            return False
        c[0] = True
        return True

    def _release(c, *, refcount=False):
        state["releases"].append(True)
        c[0] = False

    bc = types.SimpleNamespace(
        _tts_playback_active=cell,
        _pa_claim_owner=_claim,
        _pa_release_owner=_release,
        _pa_detach_play_stream=lambda s: state["detached"].append(s) or True,
        state=state,
    )
    if with_closer:
        bc._safe_close_stream = lambda s: state["closed"].append(s)
    return bc


class StingOwnershipTests(unittest.TestCase):
    def setUp(self):
        ib._STING_CLAIMED[0] = False
        self.addCleanup(lambda: ib._STING_CLAIMED.__setitem__(0, False))

    def _play(self, sd, bc=None):
        mods = {"sounddevice": sd}
        if bc is not None:
            mods["bobert_companion"] = bc
        with mock.patch.dict(sys.modules, mods), mock.patch("builtins.print"):
            return ib._play_sting_async([0.0, 0.1], 48000, output_device=3)

    def test_sting_claims_the_playback_owner_cell(self):
        bc = _fake_bc()
        sd = _FakeSd(stream=mock.Mock())
        self.assertTrue(self._play(sd, bc))
        self.assertTrue(bc._tts_playback_active[0],
                        "the gate must see an owner while the sting's "
                        "PortAudio callback is live")
        self.assertEqual(len(sd.played), 1)
        self.assertTrue(ib._STING_CLAIMED[0])

    def test_claim_timeout_skips_the_sting_entirely(self):
        # A reinit is in flight: losing the sting beats opening a stream into a
        # live sd._terminate() (that is the crash this claim prevents).
        bc = _fake_bc(claim_ok=False)
        sd = _FakeSd(stream=mock.Mock())
        self.assertFalse(self._play(sd, bc))
        self.assertEqual(sd.played, [])
        self.assertFalse(ib._STING_CLAIMED[0])

    def test_a_failed_play_releases_the_claim(self):
        bc = _fake_bc()
        sd = _FakeSd(stream=mock.Mock(), play_raises=True)
        self.assertFalse(self._play(sd, bc))
        self.assertFalse(bc._tts_playback_active[0],
                         "a claim whose stream never opened must not pin the "
                         "gate for the life of the process")
        self.assertEqual(len(bc.state["releases"]), 1)

    def test_finish_closes_detaches_then_releases_in_that_order(self):
        bc = _fake_bc()
        stream = mock.Mock()
        sd = _FakeSd(stream=stream)
        self.assertTrue(self._play(sd, bc))
        with mock.patch.dict(sys.modules,
                             {"sounddevice": sd, "bobert_companion": bc}), \
                mock.patch("builtins.print"):
            ib._finish_sting()
        self.assertEqual(bc.state["closed"], [stream],
                         "the sting's stream must actually be closed — it used "
                         "to stay open until the next sd.play() reaped it")
        self.assertEqual(bc.state["detached"], [stream],
                         "and then leave sounddevice's shared _last_callback "
                         "slot, so no later sd.play() can close it twice")
        self.assertFalse(bc._tts_playback_active[0])
        self.assertFalse(ib._STING_CLAIMED[0])

    def test_finish_releases_even_when_the_close_blows_up(self):
        bc = _fake_bc()
        stream = mock.Mock()
        sd = _FakeSd(stream=stream)
        self.assertTrue(self._play(sd, bc))
        bc._safe_close_stream = mock.Mock(side_effect=RuntimeError("wedged"))
        with mock.patch.dict(sys.modules,
                             {"sounddevice": sd, "bobert_companion": bc}), \
                mock.patch("builtins.print"):
            ib._finish_sting()
        self.assertFalse(bc._tts_playback_active[0],
                         "a stuck claim would make _refresh_devices defer "
                         "every reinit for the whole session")

    def test_finish_without_a_close_helper_closes_inline(self):
        bc = _fake_bc(with_closer=False)
        stream = mock.Mock()
        sd = _FakeSd(stream=stream)
        self.assertTrue(self._play(sd, bc))
        with mock.patch.dict(sys.modules,
                             {"sounddevice": sd, "bobert_companion": bc}), \
                mock.patch("builtins.print"):
            ib._finish_sting()
        stream.stop.assert_called_once()
        stream.close.assert_called_once()
        self.assertFalse(bc._tts_playback_active[0])

    # ── the degraded path: no monolith at all ────────────────────────────
    def test_stand_alone_run_still_plays_and_still_tears_down(self):
        sd = _FakeSd(stream=mock.Mock())
        with mock.patch.dict(sys.modules, {"sounddevice": sd}):
            sys.modules.pop("bobert_companion", None)
            with mock.patch("builtins.print"):
                self.assertTrue(ib._play_sting_async([0.0], 48000))
                ib._finish_sting()
        self.assertEqual(len(sd.played), 1)
        self.assertFalse(ib._STING_CLAIMED[0])

    def test_host_lookup_never_imports_the_monolith(self):
        # sys.modules LOOKUP, not an import: this module is self-contained by
        # design (callers pass speak_fn / write_hud_state in) and must keep
        # working stand-alone. Both assertions live INSIDE the patch.dict block
        # on purpose — it restores sys.modules on exit, so an assertion after
        # the block would be about the RESTORED dict and would fail in a
        # full-suite run purely because some earlier monolith test had already
        # imported bobert_companion.
        with mock.patch.dict(sys.modules, {}, clear=False):
            sys.modules.pop("bobert_companion", None)
            self.assertIsNone(ib._sting_host())
            self.assertNotIn("bobert_companion", sys.modules,
                             "_sting_host must not import the monolith")

    def test_host_lookup_rejects_a_monolith_without_the_gate(self):
        # An older monolith (or a SimpleNamespace double) must degrade to the
        # unguarded behaviour, never crash the boot sequence.
        half = types.SimpleNamespace(_tts_playback_active=[False])
        with mock.patch.dict(sys.modules, {"bobert_companion": half}):
            self.assertIsNone(ib._sting_host())


class StingIsFinishedByTheBootSequenceTests(unittest.TestCase):
    """The claim/close pair is worthless if play_iron_man_boot never closes it:
    the flag would stay up for the whole session and _refresh_devices would
    defer every re-enumeration forever (JARVIS stops following device changes)."""

    def test_boot_sequence_finishes_the_sting_before_speaking(self):
        order = []
        with mock.patch.object(ib, "_load_sting_from_disk",
                               return_value=([0.0], 48000)), \
                mock.patch.object(ib, "_play_sting_async",
                                  side_effect=lambda *a, **k: order.append("play") or True), \
                mock.patch.object(ib, "_finish_sting",
                                  side_effect=lambda: order.append("finish")), \
                mock.patch.object(ib.time, "sleep",
                                  side_effect=lambda s: order.append("wait")), \
                mock.patch("builtins.print"):
            ib.play_iron_man_boot(speak_fn=lambda line: order.append("speak"),
                                  write_hud_state=None)
        self.assertEqual(order[:4], ["play", "wait", "finish", "speak"],
                         "the stream must be closed and the claim dropped "
                         "before speak_fn opens its own playback stream")


if __name__ == "__main__":
    unittest.main()
