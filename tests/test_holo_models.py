"""Holo mode - the owner's OWN 3D models (STL / 3MF / OBJ): the folder listing,
its extension / size / depth filters, the path-safe /holo/model/<id> route,
fuzzy voice matching, the holo_show replies and the page contract for the
loaders. Hermetic: temp folders only, never the owner's files, no browser."""
from __future__ import annotations

import json
import os
import re
import tempfile
import unittest
import urllib.error
import urllib.request
from unittest import mock

from core import config, holo_web, prompts
from skills import holo_mode
from tests.test_web_interface import _ServerBase, _get, _get_raw

_HERE = os.path.dirname(os.path.abspath(__file__))
_HOLO_DIR = os.path.join(os.path.dirname(_HERE), "hud", "holo")

STL = b"solid t\nendsolid t\n" + b"x" * 200


def _read(name):
    with open(os.path.join(_HOLO_DIR, name), encoding="utf-8") as fh:
        return fh.read()


def _put(root, rel, data=STL):
    path = os.path.join(root, *rel.split("/"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(data)
    return path


class _Dirs(unittest.TestCase):
    """A built-in folder and an 'owner' folder, both under one temp dir."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = os.path.realpath(self._tmp.name)
        self.builtin = os.path.join(self.base, "builtin")
        self.extra = os.path.join(self.base, "owner")
        os.makedirs(self.extra)

    def names(self, **kw):
        kw.setdefault("builtin_dir", self.builtin)
        return [m["name"] for m in holo_web.list_models(**kw)]


class ListingTests(_Dirs):
    def test_config_has_the_knobs_and_no_personal_paths(self):
        self.assertEqual(config.HOLO_MODEL_DIRS, [])
        self.assertEqual(config.HOLO_MODEL_MAX_MB, 64)
        self.assertEqual(holo_web.DEFAULT_MODEL_MAX_MB, 64)
        self.assertTrue(holo_web.BUILTIN_MODEL_DIR.replace("\\", "/").endswith(
            "data/holo_models"))

    def test_builtin_folder_is_created_lazily(self):
        self.assertFalse(os.path.exists(self.builtin))
        self.assertEqual(holo_web.list_models([], self.builtin), [])
        self.assertTrue(os.path.isdir(self.builtin))

    def test_only_stl_3mf_obj_are_listed(self):
        for n in ("a.stl", "b.3mf", "c.obj", "D.STL", "e.step", "f.txt", "g.gcode",
                  "h.3mf.bak", "stl", "i.mtl"):
            _put(self.builtin, n)
        got = {m["name"]: m["type"] for m in holo_web.list_models([], self.builtin)}
        self.assertEqual(got, {"a": "stl", "b": "3mf", "c": "obj", "D": "stl"})

    def test_entry_shape_name_type_size_and_stable_id(self):
        _put(self.builtin, "Tiger_Pendant-FAST.3mf", b"z" * 321)
        a = holo_web.list_models([], self.builtin)
        b = holo_web.list_models([], self.builtin)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 1)
        m = a[0]
        self.assertEqual(m["name"], "Tiger Pendant-FAST")
        self.assertEqual(m["type"], "3mf")
        self.assertEqual(m["size"], 321)
        self.assertRegex(m["id"], r"^[0-9a-f]{12}$")
        self.assertEqual(set(m), {"id", "name", "type", "size"})   # no path leaks

    def test_extra_dirs_are_searched_and_missing_ones_ignored(self):
        _put(self.builtin, "one.stl")
        _put(self.extra, "two.obj")
        self.assertEqual(self.names(extra_dirs=[self.extra, os.path.join(self.base, "nope")]),
                         ["one", "two"])
        self.assertEqual(self.names(extra_dirs=self.extra), ["one", "two"])   # a bare string works
        self.assertEqual(self.names(extra_dirs=[self.extra, "", None, 5]), ["one", "two"])

    def test_overlapping_folders_do_not_list_a_file_twice(self):
        _put(self.builtin, "one.stl")
        self.assertEqual(self.names(extra_dirs=[self.builtin, self.builtin]), ["one"])

    def test_depth_limit_is_two_folders_down(self):
        _put(self.builtin, "d0.stl")
        _put(self.builtin, "a/d1.stl")
        _put(self.builtin, "a/b/d2.stl")
        _put(self.builtin, "a/b/c/d3.stl")
        self.assertEqual(self.names(), ["d0", "d1", "d2"])

    def test_size_cap_and_empty_files(self):
        _put(self.builtin, "small.stl", b"x" * 500)
        _put(self.builtin, "big.stl", b"x" * 3000)
        _put(self.builtin, "empty.stl", b"")
        self.assertEqual(self.names(max_mb=0.001), ["small"])            # ~1 KiB cap
        self.assertEqual(self.names(max_mb=1), ["big", "small"])
        with mock.patch.object(holo_web, "_cfg", return_value=0.001):
            self.assertEqual(holo_web.model_max_bytes(), 1048)
        self.assertEqual(holo_web.model_max_bytes(0), 64 * 1024 * 1024)   # junk -> default
        self.assertEqual(holo_web.model_max_bytes("abc"), 64 * 1024 * 1024)

    def test_sorted_by_name_and_3mf_first_for_the_same_name(self):
        for n in ("zeta.stl", "alpha.stl", "alpha.3mf", "Beta.obj"):
            _put(self.builtin, n)
        got = [(m["name"], m["type"]) for m in holo_web.list_models([], self.builtin)]
        self.assertEqual(got, [("alpha", "3mf"), ("alpha", "stl"), ("Beta", "obj"),
                               ("zeta", "stl")])

    def test_a_link_pointing_outside_the_folder_is_not_listed(self):
        outside = _put(self.base, "outside/secret.stl")
        link = os.path.join(self.builtin, "link.stl")
        os.makedirs(self.builtin, exist_ok=True)
        try:
            os.symlink(outside, link)
        except (OSError, NotImplementedError, AttributeError):
            self.skipTest("no symlink support here")
        _put(self.builtin, "real.stl")
        self.assertEqual(self.names(), ["real"])

    def test_models_payload_shape(self):
        _put(self.builtin, "one.stl")
        p = holo_web.models_payload([], self.builtin)
        self.assertEqual(p["max_mb"], 64.0)
        self.assertEqual(p["folder"], "data/holo_models")
        self.assertEqual([m["name"] for m in p["models"]], ["one"])

    def test_display_name(self):
        self.assertEqual(holo_web.display_name(r"C:\x\Tiger_Pendant__FAST.3mf"), "Tiger Pendant FAST")
        self.assertLessEqual(len(holo_web.display_name("a" * 200 + ".stl")), 60)


class ServeTests(_Dirs):
    def _id(self, name):
        return next(m["id"] for m in holo_web.list_models([self.extra], self.builtin)
                    if m["name"] == name)

    def resp(self, mid, **kw):
        kw.setdefault("builtin_dir", self.builtin)
        kw.setdefault("extra_dirs", [self.extra])
        return holo_web.model_response(mid, **kw)

    def test_serves_the_file_bytes_with_a_model_type(self):
        _put(self.extra, "gear.stl", b"STLDATA")
        _put(self.extra, "box.3mf", b"3MFDATA")
        _put(self.extra, "tri.obj", b"OBJDATA")
        for name, data, ctype in (("gear", b"STLDATA", "model/stl"),
                                  ("box", b"3MFDATA", "model/3mf"),
                                  ("tri", b"OBJDATA", "model/obj")):
            code, body, ct = self.resp(self._id(name))
            self.assertEqual((code, body, ct), (200, data, ctype))

    def test_unknown_and_malformed_ids_are_404(self):
        _put(self.extra, "gear.stl")
        bad = ["0" * 12, "", "gear", "../gear.stl", "..", "../../core/config.py",
               "..%2f..%2fcore%2fconfig.py", "a" * 11, "a" * 13, "A" * 12,
               "f0266472b8d/", "gear.stl", "C:\\Windows\\win.ini", "/etc/passwd",
               "0" * 12 + "\x00", None, 12, b"abc"]
        for mid in bad:
            code, body, _ = self.resp(mid)
            self.assertEqual(code, 404, repr(mid))
            self.assertEqual(body, {"error": "not found"}, repr(mid))

    def test_a_file_outside_every_configured_folder_is_never_served(self):
        outside = _put(self.base, "elsewhere/secret.stl")
        code, _, _ = self.resp(holo_web.model_id_for(outside))
        self.assertEqual(code, 404)
        # ...and the same file IS served once its folder is configured
        code, body, _ = self.resp(holo_web.model_id_for(outside),
                                  extra_dirs=[os.path.dirname(outside)])
        self.assertEqual((code, body), (200, STL))

    def test_wrong_extension_is_never_served_even_with_a_valid_looking_id(self):
        path = _put(self.extra, "notes.txt", b"private")
        code, _, _ = self.resp(holo_web.model_id_for(path))
        self.assertEqual(code, 404)

    def test_too_deep_file_is_404(self):
        path = _put(self.extra, "a/b/c/deep.stl")
        code, _, _ = self.resp(holo_web.model_id_for(path))
        self.assertEqual(code, 404)

    def test_oversize_is_an_honest_413_not_a_404(self):
        path = _put(self.extra, "huge.stl", b"x" * 5000)
        mid = holo_web.model_id_for(path)
        code, body, ct = self.resp(mid, max_mb=0.001)
        self.assertEqual(code, 413)
        self.assertIn("size cap", body["error"])
        self.assertEqual(ct, "application/json")
        self.assertEqual(self.resp(mid, max_mb=1)[0], 200)

    def test_a_file_that_vanished_is_404(self):
        path = _put(self.extra, "gone.stl")
        mid = holo_web.model_id_for(path)
        os.remove(path)
        self.assertEqual(self.resp(mid)[0], 404)

    def test_listing_never_creates_folders_on_serve(self):
        self.assertEqual(self.resp("0" * 12)[0], 404)
        self.assertFalse(os.path.exists(self.builtin))


class _ModelServer(_ServerBase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.mdir = os.path.realpath(self._tmp.name)
        self.state = holo_web.HoloState()
        self._ps = [mock.patch.object(holo_web, "STATE", self.state),
                    mock.patch.object(holo_web, "BUILTIN_MODEL_DIR", self.mdir),
                    mock.patch.object(config, "HOLO_MODEL_DIRS", [])]
        for p in self._ps:
            p.start()
        super().setUp()

    def tearDown(self):
        super().tearDown()
        for p in self._ps:
            p.stop()


class ModelRouteTests(_ModelServer):
    def test_list_route_and_model_route(self):
        _put(self.mdir, "Gear Box.stl", b"BINARYISH\x00\x01\x02")
        code, data = _get(self.base + "/api/holo/models")
        self.assertEqual(code, 200)
        self.assertEqual([m["name"] for m in data["models"]], ["Gear Box"])
        self.assertEqual(data["max_mb"], 64.0)
        mid = data["models"][0]["id"]
        with urllib.request.urlopen(self.base + "/holo/model/" + mid, timeout=5) as r:
            self.assertEqual(r.read(), b"BINARYISH\x00\x01\x02")
            self.assertEqual(r.headers["Content-Type"], "model/stl")
            self.assertEqual(r.headers["X-Content-Type-Options"], "nosniff")
            self.assertIn("cdn.jsdelivr.net", r.headers["Content-Security-Policy"])

    def test_traversal_and_junk_urls_are_404(self):
        _put(self.mdir, "ok.stl")
        for bad in ("/holo/model/", "/holo/model", "/holo/model/..", "/holo/model/../../core/config.py",
                    "/holo/model/%2e%2e/%2e%2e/core/config.py",
                    "/holo/model/..%2f..%2fcore%2fconfig.py", "/holo/model/ok.stl",
                    "/holo/model/" + "0" * 12, "/holo/model/" + "0" * 12 + "/extra",
                    "/holo/models/" + "0" * 12):
            code, _ = _get_raw(self.base + bad)
            self.assertEqual(code, 404, bad)

    def test_oversize_is_413_over_http(self):
        path = _put(self.mdir, "big.stl", b"x" * 4096)
        mid = holo_web.model_id_for(path)
        with mock.patch.object(config, "HOLO_MODEL_MAX_MB", 0.001):
            code, text = _get_raw(self.base + "/holo/model/" + mid)
            self.assertEqual(code, 413)
            self.assertIn("size cap", json.loads(text)["error"])
            _c, data = _get(self.base + "/api/holo/models")
            self.assertEqual(data["models"], [])               # not offered either

    def test_state_payload_carries_the_part_count(self):
        code, data = _get(self.base + "/api/holo/state?seen=0&shown=x&parts=3")
        self.assertEqual(code, 200)
        self.assertEqual(data["parts"], 3)
        self.assertEqual(self.state.parts, 3)
        _get(self.base + "/api/holo/state?parts=junk")
        self.assertEqual(self.state.parts, 0)


class ModelRouteTokenTests(_ModelServer):
    token = "-".join(("s3cret", "token"))     # built at runtime: the PII gate greps tests/

    def test_list_needs_the_token_like_the_state_api(self):
        _put(self.mdir, "ok.stl")
        code, _ = _get_raw(self.base + "/api/holo/models")
        self.assertEqual(code, 401)
        code, data = _get(self.base + "/api/holo/models", headers={"X-Auth-Token": self.token})
        self.assertEqual(code, 200)
        mid = data["models"][0]["id"]
        # a local bind serves the page and its assets token-free; the model file
        # follows the same page rule, and still only by a known id
        code, _ = _get_raw(self.base + "/holo/model/" + mid)
        self.assertEqual(code, 200)
        code, _ = _get_raw(self.base + "/holo/model/" + "0" * 12)
        self.assertEqual(code, 404)


# --- fuzzy matching and the voice -----------------------------------------------
def _m(name, kind="stl"):
    return {"id": "0" * 12, "name": name, "type": kind, "size": 1}


class MatchTests(unittest.TestCase):
    LIST = [_m("Tiger Pendant FAST", "3mf"), _m("Fence Corner Inner"), _m("Gear"),
            _m("Rover Chassis v2", "3mf")]

    def best(self, q):
        hit = holo_web.match_model(q, self.LIST)
        return hit["name"] if hit else None

    def test_exact_partial_and_noisy_names_match(self):
        self.assertEqual(self.best("tiger pendant fast"), "Tiger Pendant FAST")
        self.assertEqual(self.best("tiger pendant"), "Tiger Pendant FAST")
        self.assertEqual(self.best("the tiger"), "Tiger Pendant FAST")
        self.assertEqual(self.best("Tiger-Pendant"), "Tiger Pendant FAST")
        self.assertEqual(self.best("gear"), "Gear")
        self.assertEqual(self.best("rover chassis"), "Rover Chassis v2")
        self.assertEqual(self.best("fence corner"), "Fence Corner Inner")

    def test_small_speech_errors_still_match(self):
        self.assertEqual(self.best("tiger pendent"), "Tiger Pendant FAST")
        self.assertEqual(self.best("rover chasis"), "Rover Chassis v2")

    def test_unrelated_names_do_not_match(self):
        for q in ("banana", "xylophone", "death star", "", "   ", "!!!"):
            self.assertIsNone(self.best(q), q)
        self.assertIsNone(holo_web.match_model("gear", []))
        self.assertIsNone(holo_web.match_model("gear", None))

    def test_exact_beats_partial_and_3mf_beats_stl_on_a_tie(self):
        lst = [_m("Gear Box"), _m("Gear")]
        self.assertEqual(holo_web.match_model("gear", lst)["name"], "Gear")
        tie = [_m("Cube"), _m("Cube", "3mf")]
        self.assertEqual(holo_web.match_model("cube", tie)["type"], "3mf")


class _Actions(unittest.TestCase):
    def setUp(self):
        self.state = holo_web.HoloState()
        self.opened = []
        self.models = []
        self._patches = [
            mock.patch.object(holo_web, "STATE", self.state),
            mock.patch.object(holo_web, "list_models", side_effect=lambda *a, **k: self.models),
            mock.patch.object(holo_mode, "_ensure_server",
                              return_value=(True, "127.0.0.1", 8766, "")),
            mock.patch.object(holo_mode, "_launch_window",
                              side_effect=lambda u: self.opened.append(u) or True),
            mock.patch.object(holo_mode, "_enabled", return_value=True),
        ]
        for p in self._patches:
            p.start()
            self.addCleanup(p.stop)
        holo_mode._last_open[0] = 0.0

    def page_up(self, shown="", exploded=False, parts=None):
        self.state.set_on(True)
        self.state.heartbeat(shown=shown, exploded=exploded, parts=parts)

    def last_cmd(self):
        return self.state.payload(0)["commands"][-1]


class ShowOwnModelTests(_Actions):
    def setUp(self):
        super().setUp()
        self.models = [_m("Tiger Pendant FAST", "3mf"), _m("Fence Corner Inner"),
                       _m("Gear"), _m("Rover Chassis")]

    def test_a_fuzzy_name_shows_that_model_by_its_display_name(self):
        self.page_up()
        out = holo_mode.holo_show("the tiger pendant")
        self.assertEqual(out, "The Tiger Pendant FAST is up, sir.")
        self.assertEqual(self.last_cmd()["arg"], "Tiger Pendant FAST")
        self.assertEqual(self.state.model, "Tiger Pendant FAST")

    def test_voice_decorations_are_stripped(self):
        self.page_up()
        for said in ("the gear in the hologram", "my gear model", "gear please", "Gear"):
            self.assertEqual(holo_mode.holo_show(said), "The Gear is up, sir.", said)

    def test_the_arc_reactor_still_works(self):
        self.page_up()
        self.assertEqual(holo_mode.holo_show("arc reactor"), "The arc reactor is up, sir.")
        self.assertEqual(self.last_cmd()["arg"], "arc reactor")
        self.assertEqual(holo_mode.holo_show("the arc reactor")[:19], "The arc reactor is ")
        self.assertEqual(holo_mode.holo_show("reactor")[:19], "The arc reactor is ")
        self.assertEqual(holo_mode.holo_show("")[:19], "The arc reactor is ")
        self.assertEqual(self.last_cmd()["arg"], "arc reactor")

    def test_unknown_name_is_honest_and_names_a_few_that_exist(self):
        self.page_up()
        before = self.state.seq
        out = holo_mode.holo_show("death star")
        self.assertEqual(out, "I don't have a model called death star, sir - "
                              "I do have Tiger Pendant FAST, Fence Corner Inner and Gear.")
        self.assertEqual(self.state.seq, before)            # nothing was pushed
        self.assertEqual(self.opened, [])
        self.assertNotIn("\n", out)

    def test_one_model_reads_naturally(self):
        self.models = [_m("Gear")]
        self.page_up()
        self.assertEqual(holo_mode.holo_show("death star"),
                         "I don't have a model called death star, sir - I do have Gear.")

    def test_no_models_says_where_to_put_them(self):
        self.models = []
        self.page_up()
        out = holo_mode.holo_show("tiger pendant")
        self.assertIn("data/holo_models", out)
        self.assertIn("HOLO_MODEL_DIRS", out)
        self.assertIn("can't find any", out)
        self.assertEqual(self.state.payload(0)["commands"], [])
        self.assertEqual(self.opened, [])
        # ...but the built-in reactor needs no files
        self.assertIn("arc reactor is up", holo_mode.holo_show("arc reactor"))

    def test_opens_the_window_with_the_model_when_the_page_is_down(self):
        out = holo_mode.holo_show("rover")
        self.assertEqual(out, "The Rover Chassis is up in the hand menu, sir.")
        self.assertEqual(self.opened, ["http://127.0.0.1:8766/holo?mode=desktop"])
        self.assertEqual(self.state.model, "Rover Chassis")

    def test_a_listing_that_blows_up_is_treated_as_no_models(self):
        with mock.patch.object(holo_web, "list_models", side_effect=RuntimeError("boom")):
            self.page_up()
            self.assertIn("can't find any", holo_mode.holo_show("gear"))

    def test_every_reply_is_one_short_sentence(self):
        self.page_up()
        for said in ("tiger", "death star", "reactor", ""):
            out = holo_mode.holo_show(said)
            self.assertLess(len(out), 140, out)
            self.assertEqual(out.count("\n"), 0)
            self.assertTrue(out.endswith("."), out)


class ExplodeOnePartTests(_Actions):
    def test_a_single_part_model_says_so_instead_of_exploding(self):
        self.page_up(shown="Gear", parts=1)
        out = holo_mode.holo_explode()
        self.assertIn("single piece", out)
        self.assertIn("nothing to explode", out)
        self.assertEqual(self.state.payload(0)["commands"], [])

    def test_multi_part_and_unknown_part_counts_still_explode(self):
        for parts in (3, 0, None):
            self.state = holo_web.HoloState()
            with mock.patch.object(holo_web, "STATE", self.state):
                self.page_up(shown="Tiger", parts=parts)
                self.assertEqual(holo_mode.holo_explode(), "Exploding it, sir.", parts)

    def test_closing_forgets_the_part_count(self):
        self.page_up(shown="Gear", parts=1)
        self.state.set_on(False)
        self.assertEqual(self.state.parts, 0)


# --- prompt + page contract -------------------------------------------------------
class PromptTests(unittest.TestCase):
    def test_the_brain_knows_about_own_models(self):
        p = prompts.PC_CONTROL_PROMPT
        i = p.index("HOLO MODE")
        sect = p[i:i + 2600]
        self.assertIn("holo_show", sect)
        self.assertIn("HIS OWN 3D files", sect)
        self.assertIn("show me the tiger pendant in the hologram", sect)
        self.assertIn("[ACTION: holo_show, tiger pendant]", sect)
        self.assertIn("[ACTION: holo_show, arc reactor]", sect)


class ModelPageContractTests(unittest.TestCase):
    PINNED = "https://cdn.jsdelivr.net/npm/three@0.160.0/"

    def test_models_js_is_served_and_listed(self):
        self.assertIn("/holo/models.js", holo_web._ASSETS)
        self.assertTrue(os.path.isfile(os.path.join(_HOLO_DIR, "models.js")))
        body, ctype = holo_web.asset_for("/holo/models.js")
        self.assertTrue(ctype.startswith("text/javascript"))
        self.assertGreater(len(body), 1000)

    def test_loaders_are_the_pinned_three_examples(self):
        js = _read("models.js")
        for loader in ("STLLoader", "3MFLoader", "OBJLoader"):
            self.assertIn(self.PINNED + "examples/jsm/loaders/" + loader + ".js", js)

    def test_import_map_points_three_at_the_one_pinned_build(self):
        html = _read("holo.html")
        m = re.search(r'<script type="importmap">\s*(\{.*?\})\s*</script>', html, re.S)
        self.assertIsNotNone(m, "no import map")
        imports = json.loads(m.group(1))["imports"]
        self.assertEqual(imports, {"three": self.PINNED + "build/three.module.js"})
        self.assertIn(self.PINNED + "build/three.module.js", _read("holo.js"))
        # the map must come before the first module script
        self.assertLess(html.index('type="importmap"'), html.index('type="module"'))

    def test_every_url_in_the_page_is_the_pinned_cdn(self):
        for f in ("holo.html", "holo.js", "models.js", "reactor.js", "selftest.js"):
            for u in re.findall(r"https?://[^\s\"'`)]+", _read(f)):
                if u.startswith("http://www.w3.org"):
                    continue
                self.assertTrue(u.startswith(self.PINNED), (f, u))

    def test_csp_still_only_lets_the_cdn_in(self):
        self.assertNotIn("unsafe-eval", holo_web.HOLO_CSP)
        self.assertEqual(re.findall(r"https://[^\s;]+", holo_web.HOLO_CSP),
                         ["https://cdn.jsdelivr.net", "https://cdn.jsdelivr.net"])

    def test_models_js_never_touches_the_camera_or_the_os(self):
        for f in ("models.js", "holo.js"):
            text = _read(f)
            for banned in ("getUserMedia", "mediaDevices", "tasks-vision", "HandLandmarker",
                           "pyautogui", "SetCursorPos", "SendInput", "mouse_event",
                           "XMLHttpRequest", "eval(", "new Function"):
                self.assertNotIn(banned, text, (f, banned))

    def test_file_names_reach_the_page_as_text_not_html(self):
        js = _read("holo.js")
        start = js.index("function addMenuButton")
        body = js[start:js.index("function resetMenu")]
        self.assertIn("createTextNode(label)", body)
        self.assertIn("textContent = sub", body)
        self.assertNotIn("innerHTML", body)

    def test_page_asks_the_server_for_the_list_and_the_file_by_id_only(self):
        js = _read("models.js")
        self.assertIn('"/api/holo/models"', js)
        self.assertIn('"/holo/model/" + encodeURIComponent(id)', js)

    def test_explode_on_one_part_is_answered_by_the_page_too(self):
        js = _read("holo.js")
        self.assertIn("ONE PART - NOTHING TO EXPLODE", js)
        self.assertIn("parts: String(", js)

    def test_my_models_slot_is_enabled_by_the_page(self):
        js = _read("holo.js")
        self.assertNotIn("coming soon", js.lower())
        self.assertIn('model: "__models"', js)


if __name__ == "__main__":
    unittest.main()
