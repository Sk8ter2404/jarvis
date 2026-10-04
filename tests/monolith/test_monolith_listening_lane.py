"""The listening lane (2026-10-04): no CUDA context on the brain's card, a
device setting per listening model, the music gate, the boot devices line.

THE LIVE FINDINGS (session 2026-10-04 13:49, JARVIS pid 95748):
  * JARVIS held ~755 MB on the RTX 3090 — the card that holds the local brain
    with ~1.1 GB to spare. Its CUDA context there was born at 13:50:56.037,
    inside the ambient listener's first voice-ID: core/voice_id built
    Resemblyzer's VoiceEncoder with no device, and Resemblyzer's default is
    "cuda" = cuda:0. Four free-VRAM probes used torch.cuda.mem_get_info(i),
    which creates a context on the card it asks (+59 MB replayed on the 1650).
  * With music playing in wake-word mode every capture ran to the 30 s cap,
    Parakeet decoded it, Whisper decoded it again as a "no-wake" rescue (427
    of 536 that session), and the ambient listener ran Whisper on every 2.5 s
    of lyrics: the 1650 ~35 % busy, ~39 CPU-s a minute.

Pinned here, on the REAL monolith with fakes at the edges (no GPU, no
voiceprint, no meter — the harness stubs the music gate's meter to "no
music"; each test that needs music says so):
  * GPUs are numbered by PCI bus before anything initialises CUDA;
  * a session's probes and voice-ID put NOTHING on cuda:0 (a fake torch and a
    Resemblyzer with its real device default count every context);
  * Parakeet / Smart Turn / Whisper placement: the CPU by default (the call
    R6 shipped, byte for byte), a card only when it is there with room and a
    GPU runtime, else the CPU with ONE "[listen]" line;
  * the music gate: 'off' reads nothing; 'shadow' changes nothing but counts
    (and logs a rescue it would have lost); 'on' skips the rescue without a
    wake hint or the owner's voice, skips ambient batches that are not the
    owner's voice, cuts an owner capture at MUSIC_MAX_CAPTURE_S;
  * one "[music-gate]" line per minute with music; one "[listen] devices:"
    line at boot.

    python -m unittest tests.monolith.test_monolith_listening_lane
"""
from __future__ import annotations

import ast
import contextlib
import inspect
import io
import os
import sys
import threading
import types
import unittest
from unittest import mock

from tests._monolith_harness import MonolithGlobalsTestCase, requires_monolith

try:
    import numpy as np
except Exception:  # pragma: no cover - light CI
    np = None

GPUS = [
    {"index": 0, "name": "NVIDIA GeForce RTX 3090", "uuid": "GPU-aaaa1111",
     "pci_bus": 1, "total_mb": 24576, "free_mb": 1100, "used_mb": 23476,
     "util_pct": 17},
    {"index": 1, "name": "NVIDIA GeForce GTX 1650 SUPER",
     "uuid": "GPU-bbbb2222", "pci_bus": 8, "total_mb": 4096,
     "free_mb": 2769, "used_mb": 1327, "util_pct": 0},
]
MUSIC_STATE = {"standby": False, "wake_mode": True, "music_refuse": True,
               "room_music": False}


@requires_monolith
class _Base(MonolithGlobalsTestCase):
    def _p(self, *args, **kwargs):
        patcher = mock.patch.object(*args, **kwargs)
        m = patcher.start()
        self.addCleanup(patcher.stop)
        return m

    def _gpus(self, gpus=GPUS):
        """The card probe sees ``gpus`` (CUDA order = PCI order)."""
        self._p(self.bc._gpu_probe, "cuda_gpus", return_value=list(gpus))

    # create=True: these classes also run against origin/main (no gate
    # there), where they fail on behaviour, not on a missing name.
    def _mode(self, mode):
        self._p(self.bc, "MUSIC_GATE_MODE", mode, create=True)

    def _music(self, on=True):
        """The meter reads music (or not); only a wake-word line gets
        through (wake-word mode)."""
        self._p(self.bc, "_music_meter",
                return_value=(0.3, False) if on else (0.0, False),
                create=True)
        self._p(self.bc, "_wake_gate_state", return_value=dict(MUSIC_STATE))

    def _counts(self):
        return self.bc._music_counter.snapshot()


class CudaOrderTests(_Base):
    def test_gpus_are_numbered_by_pci_bus(self):
        self.assertEqual(os.environ.get("CUDA_DEVICE_ORDER"), "PCI_BUS_ID")

    def test_set_before_any_heavy_import(self):
        # CUDA reads it once, at its first initialisation: it must be set at
        # the top of the module, before numpy / cv2 / anything that could.
        path = inspect.getsourcefile(self.bc)
        with open(path, encoding="utf-8") as f:
            tree = ast.parse(f.read())
        setdefault_line = first_heavy = None
        for node in tree.body:
            src = ast.unparse(node)
            if (setdefault_line is None and "CUDA_DEVICE_ORDER" in src
                    and "setdefault" in src):
                setdefault_line = node.lineno
            if first_heavy is None and isinstance(node, ast.Import) and any(
                    a.name in ("numpy", "cv2", "torch", "ctranslate2")
                    for a in node.names):
                first_heavy = node.lineno
        self.assertIsNotNone(setdefault_line)
        self.assertLess(setdefault_line, first_heavy)


class NoBrainCardContextTests(_Base):
    """Everything a session runs to place or probe a listening model, against
    a torch that records each CUDA context it would open and a Resemblyzer
    with its real "cuda if available" default. Nothing may land on cuda:0."""

    def _fake_torch(self):
        contexts = []
        cuda = types.SimpleNamespace()
        cuda.is_available = lambda: True

        def mem_get_info(i=0):
            contexts.append(int(i))
            return (2000 * 2 ** 20, 4096 * 2 ** 20)
        cuda.mem_get_info = mem_get_info
        cuda.empty_cache = lambda: None
        cuda.device = lambda d: contextlib.nullcontext()
        torch = types.ModuleType("torch")
        torch.cuda = cuda
        torch.device = lambda d: d
        return torch, contexts

    def _fake_resemblyzer(self, contexts):
        mod = types.ModuleType("resemblyzer")

        class VoiceEncoder:
            def __init__(self, device=None, verbose=True):
                if device is None:            # Resemblyzer's own default
                    device = "cuda"
                d = str(device)
                if d.startswith("cuda"):
                    contexts.append(int(d.split(":")[1]) if ":" in d else 0)
                self.device = d
        mod.VoiceEncoder = VoiceEncoder
        return mod

    def test_a_session_opens_no_context_on_the_brains_card(self):
        bc = self.bc
        import core.voice_id as vid
        from core import voice_clone as vc
        torch, contexts = self._fake_torch()
        rz = self._fake_resemblyzer(contexts)
        # (hasattr / create=True: the same test runs against origin/main,
        # where it fails on the contexts list, not on a missing name.)
        nvml = hasattr(bc, "_gpu_probe")
        if nvml:
            self._gpus()
        self._p(vid, "_encoder", None)
        self._p(vid, "_encoder_device", "", create=True)
        import core.config as cfg
        self._p(cfg, "VOICE_ID_DEVICE", "cpu", create=True)  # shipped default
        with mock.patch.dict(sys.modules, {"torch": torch,
                                           "resemblyzer": rz}), \
                mock.patch("subprocess.run",
                           side_effect=FileNotFoundError("nvidia-smi")):
            plan = bc._whisper_cuda_plan(1)           # boot: Whisper's card
            free0 = bc._cuda0_free_vram_mb()          # local vision's gate
            ok = vc._free_vram_ok("cuda")              # the clone's gate
            enc = vid._load_encoder()                  # ambient voice-ID
        # origin/main: [1, 0, 0, 0] — the 1650 at boot, then three contexts
        # on cuda:0, the brain's card (the probes and voice-ID).
        self.assertEqual(contexts, [])               # no context anywhere
        self.assertEqual(enc.device, "cpu")
        self.assertEqual(plan[0], "int8")
        self.assertEqual(free0, 1100)                 # read from NVML
        self.assertFalse(ok)                          # 1,100 MB < 6 GB


class WhisperPlacementTests(_Base):
    def test_listen_is_the_listen_card(self):
        self._gpus()
        self._p(self.bc, "WHISPER_DEVICE", "listen")
        self._p(self.bc, "LISTEN_GPU", "1650")
        self.assertEqual(self.bc._resolve_whisper_device(), "cuda:1")

    def test_a_missing_listen_card_is_the_cpu_with_one_line(self):
        self._gpus(GPUS[:1])
        self._p(self.bc, "WHISPER_DEVICE", "listen")
        self._p(self.bc, "LISTEN_GPU", "cuda:1")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(self.bc._resolve_whisper_device(), "cpu")
        self.assertIn("[listen] whisper: asked listen — listen card 'cuda:1' "
                      "not found; running on the CPU", out.getvalue())

    def test_explicit_settings_are_unchanged(self):
        for setting in ("cpu", "cuda", "cuda:1"):
            self._p(self.bc, "WHISPER_DEVICE", setting)
            self.assertEqual(self.bc._resolve_whisper_device(), setting)


class OnnxPlacementTests(_Base):
    def _ort(self, cuda):
        ort = types.ModuleType("onnxruntime")
        provs = (["CUDAExecutionProvider"] if cuda else []) + [
            "AzureExecutionProvider", "CPUExecutionProvider"]
        ort.get_available_providers = lambda: list(provs)
        return ort

    def _load_parakeet(self, device, ort):
        bc = self.bc
        self._p(bc, "_stt_alt", None)
        self._p(bc, "PARAKEET_DEVICE", device)
        load = self._p(bc._stt_parakeet, "load", return_value=object())
        out = io.StringIO()
        with mock.patch.dict(sys.modules, {"onnxruntime": ort}), \
                contextlib.redirect_stdout(out):
            bc._parakeet_engine()
            bc._stt_alt = None
            bc._parakeet_engine()                  # a reload: no 2nd line
        return load, out.getvalue()

    def test_default_cpu_is_the_call_r6_shipped_and_asks_no_card(self):
        self._p(self.bc._gpu_probe, "find",
                side_effect=AssertionError("asked the card"))
        load, out = self._load_parakeet("cpu", self._ort(cuda=False))
        load.assert_called_with(self.bc.PARAKEET_MODEL_DIR,
                                self.bc.PARAKEET_THREADS)
        self.assertEqual(out, "")

    def test_listen_without_gpu_onnxruntime_is_the_cpu_logged_once(self):
        self._gpus()
        load, out = self._load_parakeet("listen", self._ort(cuda=False))
        load.assert_called_with(self.bc.PARAKEET_MODEL_DIR,
                                self.bc.PARAKEET_THREADS)
        self.assertEqual(out.count("[listen] parakeet: asked listen — "
                                   "onnxruntime has no CUDA provider"), 1)

    def test_listen_with_gpu_onnxruntime_and_room_is_the_card(self):
        self._gpus()
        load, _out = self._load_parakeet("listen", self._ort(cuda=True))
        self.assertEqual(load.call_args.kwargs["providers"],
                         [("CUDAExecutionProvider", {"device_id": 1}),
                          "CPUExecutionProvider"])

    def test_a_full_card_is_the_cpu(self):
        full = [dict(GPUS[0]), dict(GPUS[1], free_mb=1500)]  # < 1200 + 512
        self._gpus(full)
        load, out = self._load_parakeet("listen", self._ort(cuda=True))
        load.assert_called_with(self.bc.PARAKEET_MODEL_DIR,
                                self.bc.PARAKEET_THREADS)
        self.assertIn("1500 MB free < 1200 + 512 MB reserve", out)

    def test_smart_turn_session(self):
        bc = self.bc
        sess = self._p(bc._endpointing, "smart_turn_session",
                       return_value="session")
        self._p(bc, "SMART_TURN_DEVICE", "cpu")
        self.assertEqual(bc._smart_turn_session("st.onnx"), "session")
        sess.assert_called_once_with("st.onnx", providers=None)
        self._gpus()
        self._p(bc, "SMART_TURN_DEVICE", "cuda:1")
        with mock.patch.dict(sys.modules, {"onnxruntime": self._ort(True)}):
            bc._smart_turn_session("st.onnx")
        self.assertEqual(sess.call_args.kwargs["providers"][0],
                         ("CUDAExecutionProvider", {"device_id": 1}))

    def test_smart_turn_loads_through_the_placement(self):
        self.assertEqual(self.bc._eot_turn._factory.__code__.co_names[0],
                         "_smart_turn_session")


class BootLineTests(_Base):
    def test_the_devices_line(self):
        bc = self.bc
        self._gpus()
        self._p(bc, "_stt_device", "cuda:1")
        self._p(bc, "_stt_model_name", "large-v3-turbo")
        self._p(bc, "STT_ENGINE", "parakeet")
        self._p(bc, "SMART_TURN_MODE", "shadow")
        self._p(bc, "PARAKEET_DEVICE", "cpu")
        self._p(bc, "SMART_TURN_DEVICE", "cpu")
        self._p(bc, "LISTEN_GPU", "cuda:1")
        self._mode("shadow")
        import core.config as cfg
        self._p(cfg, "VOICE_ID_DEVICE", "cpu")
        line = bc._listen_devices_line()
        self.assertTrue(line.startswith(
            "[listen] devices: whisper=cuda:1 (NVIDIA GeForce GTX 1650 SUPER)"
            " [large-v3-turbo] parakeet=cpu ["), line)
        for part in ("smart-turn=cpu", "silero=cpu", "voice-id=cpu",
                     "lyric-loop=cpu",
                     "| listen card: cuda:1 NVIDIA GeForce GTX 1650 SUPER "
                     "(bus 8, GPU-bbbb2222) 2769/4096 MB free",
                     "| music gate: shadow"):
            self.assertIn(part, line)

    def test_off_models_and_no_card(self):
        bc = self.bc
        self._gpus(GPUS[:1])
        self._p(bc, "STT_ENGINE", "whisper")
        self._p(bc, "STT_SHADOW", "")
        self._p(bc, "SMART_TURN_MODE", "off")
        self._p(bc, "_stt_device", "cpu")
        line = bc._listen_devices_line()
        self.assertIn("whisper=cpu", line)
        self.assertIn("parakeet=off", line)
        self.assertIn("smart-turn=off", line)
        self.assertIn("listen card: none (listen card 'cuda:1' not found)",
                      line)

    def test_main_logs_it_once_after_the_turn_flags(self):
        src = inspect.getsource(self.bc.main)
        self.assertEqual(src.count("_log_listen_devices()"), 1)
        self.assertLess(src.index("_log_turn_flags()"),
                        src.index("_log_listen_devices()"))


class MusicStateTests(_Base):
    def test_off_reads_nothing(self):
        self._mode("off")
        read = self._p(self.bc, "_music_read_state",
                       side_effect=AssertionError("read the meter"))
        self.assertFalse(self.bc._music_now())
        read.assert_not_called()

    def test_music_mode_needs_audio_and_the_wake_word_rule(self):
        self._mode("shadow")
        self._music(on=True)
        self.assertTrue(self.bc._music_now())
        self.bc._music_state["at"] = None
        self.bc._music_meter.return_value = (0.0, False)
        self.assertFalse(self.bc._music_now())
        self.bc._music_state["at"] = None
        self.bc._music_meter.return_value = (0.3, False)
        self.bc._wake_gate_state.return_value = dict(
            MUSIC_STATE, wake_mode=False, music_refuse=False)
        self.assertFalse(self.bc._music_now())       # he talks normally

    def test_room_music_counts_and_an_unreadable_meter_asks_the_session(self):
        self._mode("shadow")
        self._music(on=False)
        self.bc._wake_gate_state.return_value = dict(MUSIC_STATE,
                                                     room_music=True)
        self.assertTrue(self.bc._music_now())

    def test_cached_for_two_seconds(self):
        self._mode("shadow")
        self._music(on=True)
        self.bc._music_now()
        self.bc._music_now()
        self.assertEqual(self.bc._music_meter.call_count, 1)

    def test_a_failing_read_is_no_music(self):
        self._mode("on")
        self._p(self.bc, "_music_meter", side_effect=OSError("COM"))
        self.assertFalse(self.bc._music_now())


class RescueGateTests(_Base):
    """The rescue through the REAL core/stt_parakeet.Primary instance."""
    W_TEXT = "Jarvis, what's new?"

    def setUp(self):
        bc = self.bc
        self._p(bc, "STT_ENGINE", "parakeet")
        self._p(bc, "STT_SHADOW", "")
        self._p(bc, "TURN_TAIL_PROBE", False)
        self._p(bc, "_SPECULATIVE_STT", False)
        self._p(bc, "_tail_probe_start", lambda audio: None)
        self._p(bc, "STT_REPLACEMENTS", {})
        self._p(bc, "STT_REPLACEMENTS_PARAKEET", {})
        self._p(bc._parakeet_latch, "failed", "")
        self.whisper = self._p(
            bc, "transcribe",
            return_value=(self.W_TEXT, {"no_speech_prob": 0.0,
                                        "avg_logprob": -0.2}))
        self._p(bc, "_parakeet_rescue", return_value="no-wake")
        self.notes = []
        self._p(bc, "_tt_note_stat",
                side_effect=lambda k, v: self.notes.append((k, v)))
        self.audio = np.zeros(16000 * 3, dtype=np.float32)

    def _run(self, parakeet_text, voice="not_owner"):
        bc = self.bc
        self._p(bc, "_parakeet_decode", return_value=(
            parakeet_text, {"no_speech_prob": 0.0, "avg_logprob": -0.4}))
        self.voice = self._p(bc, "_capture_voice_verdict",
                             return_value=voice, create=True)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            res = bc._transcribe_capture(self.audio)
        return res, out.getvalue()

    def test_on_over_music_without_a_hint_skips_the_rescue(self):
        self._mode("on")
        self._music()
        res, _out = self._run("la la la love you")
        self.whisper.assert_not_called()
        self.assertEqual(res[0], "la la la love you")
        self.assertIn(("stt_engine", "parakeet-gated"), self.notes)
        self.assertEqual(self._counts()["skip_rescue"], 1)
        self.assertEqual(self._counts()["rescue"], 0)

    def test_a_jarvis_like_word_rescues_without_a_voice_check(self):
        self._mode("on")
        self._music()
        res, _out = self._run("Travis, what's new")
        self.whisper.assert_called_once()
        self.voice.assert_not_called()
        self.assertEqual(res[0], self.W_TEXT)

    def test_the_owners_voice_rescues(self):
        self._mode("on")
        self._music()
        for voice in ("owner", "unsure", "unavailable"):
            self.whisper.reset_mock()
            self._run("la la la", voice=voice)
            self.whisper.assert_called_once()

    def test_no_music_rescues_as_always(self):
        self._mode("on")
        self._music(on=False)
        self._run("la la la")
        self.whisper.assert_called_once()
        self.voice.assert_not_called()

    def test_off_reads_nothing_and_rescues(self):
        self._mode("off")
        read = self._p(self.bc, "_music_read_state",
                       side_effect=AssertionError("read the meter"))
        self._run("la la la")
        self.whisper.assert_called_once()
        read.assert_not_called()
        self.assertEqual(self._counts()["rescue"], 0)   # off counts nothing

    def test_shadow_rescues_and_logs_a_line_it_would_have_lost(self):
        self._mode("shadow")
        self._music()
        res, out = self._run("la la la")
        self.whisper.assert_called_once()
        self.assertEqual(res[0], self.W_TEXT)
        c = self._counts()
        self.assertEqual((c["rescue"], c["would_rescue"], c["lost_rescue"]),
                         (1, 1, 1))
        self.assertIn("[music-gate] shadow: a rescue 'on' would have skipped "
                      f"(no-wake) made a line the wake gates pass "
                      f"({len(self.W_TEXT)} chars)", out)
        self.assertNotIn("what's new", out)               # never the words

    def test_shadow_lyrics_lose_nothing(self):
        self._mode("shadow")
        self._music()
        self.whisper.return_value = ("baby baby oh", {"no_speech_prob": 0.0,
                                                      "avg_logprob": -0.3})
        _res, out = self._run("la la la")
        self.assertEqual(self._counts()["lost_rescue"], 0)
        self.assertNotIn("would have lost", out)


class CaptureVoiceMemoTests(_Base):
    def test_the_rescue_check_is_reused_by_the_media_gate(self):
        bc = self.bc
        import core.voice_id as vid
        raw = np.ones(16000 * 4, dtype=np.float32) * 0.05
        gained = raw * 4.0
        self._p(bc, "_last_capture_audio", raw)
        self._p(bc, "_last_capture_sr", 16000)
        self._p(vid, "list_enrolled", return_value=["owner"])
        self._p(vid, "is_available", return_value=True)
        ident = self._p(vid, "identify_speaker", return_value=("owner", 0.8))
        self.assertEqual(bc._capture_voice_verdict(gained), "owner")
        self.assertIs(ident.call_args.args[0], raw)        # the RAW capture
        v, s = bc._learn_voice_verdict(raw, 16000, reject_below=0.45,
                                       any_enrolled=True)
        self.assertEqual((v, s), ("owner", 0.8))
        self.assertEqual(ident.call_count, 1)              # memo, no re-embed
        bc._learn_voice_verdict(raw.copy(), 16000)         # another buffer
        self.assertEqual(ident.call_count, 2)


class AmbientGateTests(_Base):
    """Over music, 'on', the ambient listener transcribes nothing and spends
    no voice-ID on it: voice-ID cannot pick the owner out of music (10-04
    13:53-17:00, 1,917 ambient lines over music: half scored >= 0.45 against
    his voiceprint, 5.8 % >= 0.72; his commands over media 0.46-0.52)."""

    def setUp(self):
        import core.voice_id as vid
        self.ident = self._p(vid, "identify_speaker",
                             side_effect=AssertionError("voice-ID ran"))
        self.batch = np.ones(40000, dtype=np.float32) * 0.1

    def test_off_and_no_music_are_none(self):
        self._mode("off")
        self.assertIsNone(self.bc._music_gate_ambient(self.batch, 16000))
        self._mode("on")
        self._music(on=False)
        self.assertIsNone(self.bc._music_gate_ambient(self.batch, 16000))

    def test_on_over_music_skips_with_no_voice_id(self):
        self._mode("on")
        self._music()
        self.assertEqual(self.bc._music_gate_ambient(self.batch, 16000),
                         {"verdict": "skip"})
        self.assertEqual(self._counts()["skip_ambient"], 1)
        self.ident.assert_not_called()

    def test_shadow_transcribes_and_counts_what_on_would_not(self):
        self._mode("shadow")
        self._music()
        gate = self.bc._music_gate_ambient(self.batch, 16000)
        self.assertEqual(gate, {"verdict": "shadow"})
        done = self.bc._music_gate_ambient_done
        done(gate, kept=True, wake=True, speaker=(None, 0.31))
        done(gate, kept=False)
        done(gate, kept=True, wake=False, speaker=("owner", 0.8))
        done(None, kept=True)                       # not a shadow batch
        c = self._counts()
        self.assertEqual((c["would_ambient"], c["lost_ambient"],
                          c["lost_ambient_wake"], c["lost_ambient_named"]),
                         (3, 2, 1, 1))
        self.assertEqual(c["skip_ambient"], 0)
        self.ident.assert_not_called()


class CaptureCutTests(_Base):
    """The REAL record_speech over a fake mic (the harness of
    tests/test_speculative_stt): 15 s of loud audio, then silence."""

    def _capture(self, seconds=15.0):
        from tests import test_speculative_stt as _spec
        n = int(seconds * 16000 / _spec._CHUNK)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            audio, _ = _spec.SpeculativeRealCaptureLoopTests._run_capture(
                self, [0.05] * n, tail_silence=30)
        return audio, out.getvalue()

    def test_on_over_music_stops_at_the_cap(self):
        self._mode("on")
        self._music()
        self._p(self.bc, "MUSIC_MAX_CAPTURE_S", 10.0, create=True)
        audio, out = self._capture()
        secs = len(audio) / 16000.0
        self.assertGreater(secs, 10.0)
        self.assertLess(secs, 11.5)                  # cap + the pre-roll
        self.assertIn("[music-gate] music playing — capture stopped at 10s",
                      out)
        self.assertEqual(self._counts()["skip_capture"], 1)

    def test_shadow_records_it_all_and_counts(self):
        self._mode("shadow")
        self._music()
        audio, out = self._capture()
        self.assertGreater(len(audio) / 16000.0, 15.0)
        self.assertNotIn("capture stopped", out)
        self.assertEqual(self._counts()["would_capture"], 1)

    def test_on_without_music_records_it_all(self):
        self._mode("on")
        self._music(on=False)
        audio, _out = self._capture()
        self.assertGreater(len(audio) / 16000.0, 15.0)

    def test_off_never_asks(self):
        self._mode("off")
        read = self._p(self.bc, "_music_read_state",
                       side_effect=AssertionError("read the meter"))
        audio, _out = self._capture()
        self.assertGreater(len(audio) / 16000.0, 15.0)
        read.assert_not_called()


class MinuteLineTests(_Base):
    def test_one_line_per_minute_with_music_and_whisper_counted_by_lane(self):
        bc = self.bc
        now = [5000.0]
        self._p(bc, "_music_counter",
                bc._music_gate.MinuteCounter(clock=lambda: now[0]))
        self._mode("shadow")
        self._music()
        self._p(bc, "_transcribe_impl",
                return_value=("x", {"no_speech_prob": 0.0}))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            bc.transcribe(np.zeros(1600, dtype=np.float32))     # main thread
            t = threading.Thread(target=bc.transcribe,
                                 args=(np.zeros(1600, dtype=np.float32),),
                                 name="ambient-listen")
            t.start()
            t.join(5)
            self.assertTrue(bc._music_now())
            now[0] += 61
            bc._music_tick()
        lines = [ln for ln in out.getvalue().splitlines()
                 if "[music-gate]" in ln]
        self.assertEqual(len(lines), 1, out.getvalue())
        self.assertIn("whisper 2 (ambient 1, turns 1, other 0", lines[0])

    def test_voice_id_calls_are_folded_in(self):
        bc = self.bc
        import core.voice_id as vid
        now = [5000.0]
        self._p(bc, "_music_counter",
                bc._music_gate.MinuteCounter(clock=lambda: now[0]))
        self._mode("shadow")
        self._p(vid, "identify_calls", 10)
        bc._music_tick()
        vid.identify_calls = 13
        bc._music_tick()
        self.assertEqual(self._counts()["voice_id"], 3)


if __name__ == "__main__":
    unittest.main()
