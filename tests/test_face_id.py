"""Tests for audio/face_id — the OpenCV face-recognition ENGINE.

No real model download, no real data/ writes, no hardware. Every cv2 touch is
replaced with an INJECTED fake detector/recognizer (set on the module's cached
singletons), and urllib is mocked for the download-readiness tests. The
enrollment store + model dir are repointed at a tempfile so the real
data/face_enroll.json and data/models/ are NEVER read or written.

Covers:
  * cosine math (numpy fallback): identical=1, orthogonal=0, opposite=-1, the
    public cosine() preferring rec.match() when a recognizer exposes one.
  * detect/embed with a fake detector+recognizer; _largest_face picks the
    biggest bbox.
  * recognize(): an embedding matching an enrolled one within threshold → that
    name; below threshold → "unknown"; multiple enrolled people → best wins.
  * enroll(): writes a TMP store, accumulates embeddings across calls, needs
    >=1 good capture, skips no-face frames.
  * _models_ready(): mock urllib to "download" to a tmp path; missing models →
    (False, reason) without raising; cached models skip re-download.
  * list_enrolled / forget round-trip.

stdlib unittest + mock only (no pytest); App-Control-safe. numpy is a light CI
dep (present), so the math runs on CI; cv2 is never imported here.
"""
from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

import numpy as np

from audio import face_id as fi


# ─── fakes (stand in for the cv2 detector / recognizer) ────────────────────

class _Frame:
    """A synthetic BGR frame: carries an embedding the FakeRec maps it to, plus
    a .shape (detect_faces reads h, w) and a .copy()."""
    def __init__(self, emb, shape=(200, 200, 3)):
        self.emb = np.asarray(emb, dtype=np.float32)
        self.shape = shape

    def copy(self):
        return self


class _FakeDet:
    """A YuNet stand-in. Returns `faces` (an Nx15 ndarray) regardless of frame,
    so a test can pin how many faces 'appear' and their bboxes."""
    def __init__(self, faces):
        self._faces = faces

    def setInputSize(self, _sz):
        pass

    def detect(self, _bgr):
        if self._faces is None:
            return False, None
        return True, self._faces


class _FakeRec:
    """An SFace stand-in. alignCrop tags the frame; feature returns that frame's
    embedding. Deliberately has NO match() so cosine() falls through to the
    numpy cosine (the testable path). A separate subclass adds match()."""
    def __init__(self):
        self._by_id = {}

    def register(self, frame):
        self._by_id[id(frame)] = frame.emb
        return frame

    def alignCrop(self, bgr, _face):
        # Remember which frame this was so feature() can return its embedding.
        self._by_id.setdefault(id(bgr), getattr(bgr, "emb", None))
        return ("aligned", id(bgr))

    def feature(self, aligned):
        emb = self._by_id.get(aligned[1])
        if emb is None:
            emb = np.zeros(4, dtype=np.float32)
        return np.asarray(emb, dtype=np.float32).reshape(1, -1)


def _face_row(x, y, w, h, score=0.99):
    """A length-15 YuNet face row: bbox + 5 landmarks (zeros) + score."""
    return np.array([x, y, w, h] + [0.0] * 10 + [score], dtype=np.float32)


def _faces(*rows):
    return np.array(list(rows), dtype=np.float32)


class FaceIdBase(unittest.TestCase):
    """Repoint the store + model dir at a tmp dir and reset the cached
    singletons per test, so nothing touches real data/ and tests don't leak
    state into each other."""

    def setUp(self):
        self._td = tempfile.mkdtemp(prefix="faceid_test_")
        self._p_enroll = mock.patch.object(
            fi, "ENROLL_PATH", os.path.join(self._td, "face_enroll.json"))
        self._p_models = mock.patch.object(
            fi, "MODELS_DIR", os.path.join(self._td, "models"))
        self._p_det = mock.patch.object(fi, "_detector", [None])
        self._p_rec = mock.patch.object(fi, "_recognizer", [None])
        self._p_log = mock.patch.object(fi, "_logged_fetch", [False])
        for p in (self._p_enroll, self._p_models, self._p_det,
                  self._p_rec, self._p_log):
            p.start()
            self.addCleanup(p.stop)

    def _inject(self, det_faces=None, rec=None):
        """Install a fake detector (returning det_faces) + recognizer on the
        cached singletons so detect/embed/recognize skip cv2 + model download.
        Returns the recognizer so a test can pre-register frame embeddings."""
        fi._detector[0] = _FakeDet(det_faces)
        rec = rec if rec is not None else _FakeRec()
        fi._recognizer[0] = rec
        return rec


# ─── cosine math ────────────────────────────────────────────────────────────
class CosineTests(FaceIdBase):
    def test_identical_is_one(self):
        a = np.array([1.0, 2.0, 3.0])
        self.assertAlmostEqual(fi._numpy_cosine(a, a), 1.0, places=6)

    def test_orthogonal_is_zero(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        self.assertAlmostEqual(fi._numpy_cosine(a, b), 0.0, places=6)

    def test_opposite_is_minus_one(self):
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        self.assertAlmostEqual(fi._numpy_cosine(a, b), -1.0, places=6)

    def test_zero_vector_is_minus_one(self):
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        self.assertEqual(fi._numpy_cosine(a, b), -1.0)

    def test_public_cosine_uses_numpy_when_no_recognizer(self):
        fi._recognizer[0] = None
        a = np.array([1.0, 1.0])
        b = np.array([1.0, 1.0])
        self.assertAlmostEqual(fi.cosine(a, b), 1.0, places=6)

    def test_public_cosine_none_inputs(self):
        self.assertEqual(fi.cosine(None, np.array([1.0])), -1.0)
        self.assertEqual(fi.cosine(np.array([1.0]), None), -1.0)

    def test_public_cosine_prefers_recognizer_match(self):
        # A recognizer exposing match() is used verbatim (the score is whatever
        # OpenCV would return). We stub cv2 in sys.modules so the import inside
        # cosine() resolves, and assert our match() value is returned.
        class _RecWithMatch:
            def match(self, a, b, flag):
                return 0.777
        fi._recognizer[0] = _RecWithMatch()
        fake_cv2 = mock.MagicMock()
        fake_cv2.FaceRecognizerSF_FR_COSINE = 0
        with mock.patch.dict("sys.modules", {"cv2": fake_cv2}):
            score = fi.cosine(np.array([1.0, 0.0]), np.array([0.0, 1.0]))
        self.assertAlmostEqual(score, 0.777, places=6)


# ─── detect / embed / largest-face ──────────────────────────────────────────
class DetectEmbedTests(FaceIdBase):
    def test_detect_returns_rows(self):
        self._inject(det_faces=_faces(_face_row(0, 0, 50, 50),
                                      _face_row(10, 10, 80, 80)))
        rows = fi.detect_faces(_Frame([1.0, 0, 0, 0]))
        self.assertEqual(len(rows), 2)

    def test_detect_none_frame_empty(self):
        self._inject(det_faces=_faces(_face_row(0, 0, 50, 50)))
        self.assertEqual(fi.detect_faces(None), [])

    def test_detect_no_detector_empty(self):
        # No injection → _get_detector tries to download models → fails (urllib
        # not reachable in test) → returns []. We force the model gate closed.
        with mock.patch.object(fi, "_models_ready", return_value=(False, "no")):
            self.assertEqual(fi.detect_faces(_Frame([1.0])), [])

    def test_largest_face_picks_biggest_bbox(self):
        small = _face_row(0, 0, 30, 30)
        big = _face_row(5, 5, 90, 90)
        mid = _face_row(2, 2, 60, 60)
        chosen = fi._largest_face([small, big, mid])
        self.assertEqual(fi._face_area(chosen), 90 * 90)

    def test_largest_face_empty_none(self):
        self.assertIsNone(fi._largest_face([]))

    def test_embed_returns_vector(self):
        rec = self._inject(det_faces=_faces(_face_row(0, 0, 50, 50)))
        f = _Frame([1.0, 2.0, 3.0, 4.0])
        rec.register(f)
        row = fi.detect_faces(f)[0]
        emb = fi.embed(f, row)
        self.assertIsNotNone(emb)
        self.assertEqual(np.asarray(emb).reshape(-1).shape[0], 4)

    def test_embed_no_recognizer_none(self):
        with mock.patch.object(fi, "_models_ready", return_value=(False, "no")):
            self.assertIsNone(fi.embed(_Frame([1.0]), _face_row(0, 0, 10, 10)))

    def test_bbox_of_rounds(self):
        self.assertEqual(fi._bbox_of(_face_row(1.4, 2.6, 30.5, 40.5)),
                         [1, 3, 30, 40])


# ─── recognize ──────────────────────────────────────────────────────────────
class RecognizeTests(FaceIdBase):
    def _enrolled_owner(self, rec, emb):
        """Directly seed the store with one enrolled person 'owner' carrying
        `emb` (bypassing the capture path)."""
        store = {"people": [{"name": "owner",
                             "embeddings": [[float(x) for x in emb]],
                             "ts": 0.0}]}
        fi._save_store(store)

    def test_match_within_threshold_named(self):
        owner_emb = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        rec = self._inject(det_faces=_faces(_face_row(0, 0, 80, 80)))
        self._enrolled_owner(rec, owner_emb)
        frame = _Frame(owner_emb)
        rec.register(frame)
        out = fi.recognize(frame)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["name"], "owner")
        self.assertGreaterEqual(out[0]["score"], fi._match_threshold())
        self.assertEqual(out[0]["bbox"], [0, 0, 80, 80])

    def test_below_threshold_unknown(self):
        owner_emb = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        stranger = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)  # cos 0
        rec = self._inject(det_faces=_faces(_face_row(0, 0, 80, 80)))
        self._enrolled_owner(rec, owner_emb)
        frame = _Frame(stranger)
        rec.register(frame)
        out = fi.recognize(frame)
        self.assertEqual(out[0]["name"], "unknown")

    def test_multiple_enrolled_best_wins(self):
        alice = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        bob = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        rec = self._inject(det_faces=_faces(_face_row(0, 0, 80, 80)))
        store = {"people": [
            {"name": "alice", "embeddings": [[1.0, 0.0, 0.0]], "ts": 0.0},
            {"name": "bob", "embeddings": [[0.0, 1.0, 0.0]], "ts": 0.0},
        ]}
        fi._save_store(store)
        # A probe close to bob (but not identical) should resolve to bob.
        probe_emb = np.array([0.1, 0.95, 0.0], dtype=np.float32)
        frame = _Frame(probe_emb)
        rec.register(frame)
        out = fi.recognize(frame)
        self.assertEqual(out[0]["name"], "bob")
        # sanity: alice was the wrong answer
        self.assertNotEqual(out[0]["name"], "alice")
        # unused vars referenced to keep intent obvious
        self.assertTrue(alice.size and bob.size)

    def test_no_face_empty(self):
        self._inject(det_faces=None)  # detector reports no faces
        self.assertEqual(fi.recognize(_Frame([1.0])), [])

    def test_no_enrolled_all_unknown(self):
        rec = self._inject(det_faces=_faces(_face_row(0, 0, 80, 80)))
        frame = _Frame(np.array([1.0, 0.0], dtype=np.float32))
        rec.register(frame)
        out = fi.recognize(frame)
        self.assertEqual(out[0]["name"], "unknown")


# ─── enroll / list / forget ─────────────────────────────────────────────────
class EnrollTests(FaceIdBase):
    def test_enroll_writes_tmp_store_and_counts(self):
        rec = self._inject(det_faces=_faces(_face_row(0, 0, 80, 80)))
        f1 = _Frame([1.0, 0.0, 0.0]); rec.register(f1)
        f2 = _Frame([0.9, 0.1, 0.0]); rec.register(f2)
        n = fi.enroll("owner", [f1, f2])
        self.assertEqual(n, 2)
        # Real store written to the TMP path (NOT real data/).
        self.assertTrue(os.path.isfile(fi.ENROLL_PATH))
        self.assertEqual(fi.list_enrolled(), [{"name": "owner", "count": 2}])

    def test_enroll_accumulates_across_calls(self):
        rec = self._inject(det_faces=_faces(_face_row(0, 0, 80, 80)))
        f1 = _Frame([1.0, 0.0]); rec.register(f1)
        fi.enroll("owner", [f1])
        f2 = _Frame([0.0, 1.0]); rec.register(f2)
        fi.enroll("owner", [f2])
        self.assertEqual(fi.list_enrolled(), [{"name": "owner", "count": 2}])

    def test_enroll_needs_at_least_one_good_capture(self):
        # Detector reports NO faces → 0 captures, store not created.
        self._inject(det_faces=None)
        n = fi.enroll("owner", [_Frame([1.0]), _Frame([2.0])])
        self.assertEqual(n, 0)
        self.assertFalse(os.path.isfile(fi.ENROLL_PATH))

    def test_enroll_skips_noface_frames_but_keeps_good(self):
        # Two frames; the detector returns a face for BOTH calls (it's stateless
        # here), but one frame is None and must be skipped without raising.
        rec = self._inject(det_faces=_faces(_face_row(0, 0, 80, 80)))
        good = _Frame([1.0, 0.0]); rec.register(good)
        n = fi.enroll("owner", [None, good])
        self.assertEqual(n, 1)

    def test_enroll_blank_name_zero(self):
        self._inject(det_faces=_faces(_face_row(0, 0, 80, 80)))
        self.assertEqual(fi.enroll("   ", [_Frame([1.0])]), 0)

    def test_forget_removes_person(self):
        fi._save_store({"people": [{"name": "owner", "embeddings": [[1.0]],
                                    "ts": 0.0}]})
        self.assertTrue(fi.forget("owner"))
        self.assertEqual(fi.list_enrolled(), [])

    def test_forget_missing_false(self):
        fi._save_store({"people": []})
        self.assertFalse(fi.forget("nobody"))

    def test_forget_is_case_insensitive(self):
        fi._save_store({"people": [{"name": "Owner", "embeddings": [[1.0]],
                                    "ts": 0.0}]})
        self.assertTrue(fi.forget("owner"))

    def test_load_store_missing_is_empty(self):
        # No file written → fresh empty store, no raise.
        self.assertEqual(fi._load_store(), {"people": []})

    def test_load_store_corrupt_is_empty(self):
        os.makedirs(os.path.dirname(fi.ENROLL_PATH), exist_ok=True)
        with open(fi.ENROLL_PATH, "w", encoding="utf-8") as fh:
            fh.write("{not json")
        self.assertEqual(fi._load_store(), {"people": []})


# ─── enroll_unknown (guest path: capture only the nearest UNKNOWN face) ──────
class _FakeRecPerFace:
    """Like _FakeRec but maps EACH face (by its bbox x) to its own embedding, so
    multiple faces in ONE frame embed to DIFFERENT vectors. alignCrop sees the
    face array (recognize/enroll pass it through _as_face_array), so we key the
    embedding off face[0] (the x coord). This lets a single frame hold a known
    owner AND an unknown stranger at once — exactly the bug scenario."""
    def __init__(self, by_x):
        self._by_x = {float(k): np.asarray(v, dtype=np.float32)
                      for k, v in by_x.items()}

    def alignCrop(self, _bgr, face):
        x = float(np.asarray(face, dtype=np.float32).reshape(-1)[0])
        return ("aligned", x)

    def feature(self, aligned):
        emb = self._by_x.get(aligned[1])
        if emb is None:
            emb = np.zeros(4, dtype=np.float32)
        return np.asarray(emb, dtype=np.float32).reshape(1, -1)


class EnrollUnknownTests(FaceIdBase):
    # Distinct, near-orthogonal embeddings so cosine cleanly separates them.
    OWNER = [1.0, 0.0, 0.0, 0.0]
    STRANGER = [0.0, 1.0, 0.0, 0.0]

    def _seed_owner(self):
        fi._save_store({"people": [{"name": "owner",
                                    "embeddings": [list(self.OWNER)],
                                    "ts": 0.0}]})

    def test_owner_closest_plus_unknown_enrolls_the_unknown(self):
        """THE regression: the owner is the LARGEST face in frame and a stranger
        is a smaller face. learn_guest('Sam') must enrol the STRANGER, never the
        owner — and Sam's stored embedding must be the stranger's, not the
        owner's (so recognition isn't corrupted)."""
        self._seed_owner()
        # Owner big (x=0, 120x120) and closest; stranger smaller (x=200, 60x60).
        owner_face = _face_row(0, 0, 120, 120)
        stranger_face = _face_row(200, 0, 60, 60)
        fi._detector[0] = _FakeDet(_faces(owner_face, stranger_face))
        fi._recognizer[0] = _FakeRecPerFace({0.0: self.OWNER,
                                             200.0: self.STRANGER})

        res = fi.enroll_unknown("Sam", [_Frame(self.OWNER)])  # frame emb unused

        self.assertEqual(res["added"], 1)
        self.assertTrue(res["saw_face"])
        self.assertTrue(res["saw_unknown"])
        # Sam is now enrolled — owner untouched (still exactly one embedding).
        names = {p["name"]: p for p in fi._load_store()["people"]}
        self.assertIn("Sam", names)
        self.assertEqual(len(names["owner"]["embeddings"]), 1)
        # Sam's stored embedding is the STRANGER's, NOT the owner's.
        sam_emb = names["Sam"]["embeddings"][0]
        self.assertAlmostEqual(fi._numpy_cosine(sam_emb, self.STRANGER), 1.0,
                               places=5)
        self.assertLess(fi._numpy_cosine(sam_emb, self.OWNER), 0.5)

    def test_only_owner_visible_enrolls_nothing(self):
        """If the ONLY face is the (recognised) owner, enrol nobody and report
        saw_face but not saw_unknown — the skill then says 'already known'
        instead of mis-storing the owner as the guest."""
        self._seed_owner()
        fi._detector[0] = _FakeDet(_faces(_face_row(0, 0, 120, 120)))
        fi._recognizer[0] = _FakeRecPerFace({0.0: self.OWNER})

        res = fi.enroll_unknown("Sam", [_Frame(self.OWNER)])

        self.assertEqual(res["added"], 0)
        self.assertTrue(res["saw_face"])
        self.assertFalse(res["saw_unknown"])
        # Nobody named Sam was stored; owner left intact.
        self.assertEqual(fi.list_enrolled(), [{"name": "owner", "count": 1}])

    def test_no_face_reports_no_face(self):
        fi._detector[0] = _FakeDet(None)  # detector sees nothing
        fi._recognizer[0] = _FakeRecPerFace({})
        res = fi.enroll_unknown("Sam", [_Frame(self.STRANGER)])
        self.assertEqual(res, {"added": 0, "saw_face": False,
                               "saw_unknown": False})

    def test_blank_name_zero(self):
        self.assertEqual(fi.enroll_unknown("  ", [_Frame(self.STRANGER)]),
                         {"added": 0, "saw_face": False, "saw_unknown": False})

    def test_largest_unknown_skips_known_picks_bigger_stranger(self):
        """_largest_unknown_face ignores recognised faces and, among the
        unknowns, returns the biggest (nearest) one."""
        known = _face_row(0, 0, 200, 200)      # biggest overall, but KNOWN
        small_unknown = _face_row(10, 10, 40, 40)
        big_unknown = _face_row(300, 0, 90, 90)
        faces = [known, small_unknown, big_unknown]
        results = [{"name": "owner"}, {"name": "unknown"}, {"name": "unknown"}]
        chosen = fi._largest_unknown_face(faces, results)
        self.assertEqual(fi._face_area(chosen), 90 * 90)

    def test_largest_unknown_all_known_is_none(self):
        faces = [_face_row(0, 0, 80, 80), _face_row(90, 0, 60, 60)]
        results = [{"name": "owner"}, {"name": "Tony"}]
        self.assertIsNone(fi._largest_unknown_face(faces, results))

    def test_largest_unknown_unpaired_result_treated_unknown(self):
        # A face with no paired recognize() entry is treated as unknown.
        faces = [_face_row(0, 0, 80, 80)]
        self.assertIsNotNone(fi._largest_unknown_face(faces, []))


# ─── model readiness (download mocked; never hits the network) ─────────────
class ModelReadinessTests(FaceIdBase):
    def _write_fake_model(self, path, size=fi._MIN_MODEL_BYTES + 10):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as fh:
            fh.write(b"\x00" * size)

    def test_cached_models_skip_download(self):
        # Both models already present (big enough) → ready without any urllib.
        self._write_fake_model(fi._yunet_path())
        self._write_fake_model(fi._sface_path())
        with mock.patch("urllib.request.urlopen") as urlopen:
            ok, reason = fi._models_ready()
        self.assertTrue(ok, reason)
        urlopen.assert_not_called()

    def test_download_success_writes_models(self):
        big = b"\x00" * (fi._MIN_MODEL_BYTES + 100)

        class _Resp:
            def __init__(self, data): self._d = data
            def read(self): return self._d
            def __enter__(self): return self
            def __exit__(self, *a): return False

        with mock.patch("urllib.request.urlopen",
                        return_value=_Resp(big)) as urlopen:
            ok, reason = fi._models_ready()
        self.assertTrue(ok, reason)
        # Two models fetched, both now on the TMP disk.
        self.assertEqual(urlopen.call_count, 2)
        self.assertTrue(fi._file_ok(fi._yunet_path()))
        self.assertTrue(fi._file_ok(fi._sface_path()))

    def test_download_failure_returns_reason_no_raise(self):
        with mock.patch("urllib.request.urlopen",
                        side_effect=OSError("network down")):
            ok, reason = fi._models_ready()
        self.assertFalse(ok)
        self.assertIn("could not download", reason)
        # Nothing was written.
        self.assertFalse(fi._file_ok(fi._yunet_path()))

    def test_download_too_small_rejected(self):
        tiny = b"<html>error</html>"

        class _Resp:
            def read(self): return tiny
            def __enter__(self): return self
            def __exit__(self, *a): return False

        with mock.patch("urllib.request.urlopen", return_value=_Resp()):
            ok, reason = fi._models_ready()
        self.assertFalse(ok)

    def test_is_available_reports_model_reason(self):
        # cv2 may or may not be importable on this host; either way is_available
        # surfaces a string reason and never raises when models are absent.
        with mock.patch.object(fi, "_models_ready",
                               return_value=(False, "models absent")):
            ok, reason = fi.is_available()
        self.assertFalse(ok)
        self.assertIsInstance(reason, str)


if __name__ == "__main__":   # pragma: no cover
    unittest.main()
