"""core.peer_memory - one memory across the owner's machines (2026-10-19:
"if I ask the desktop what we did on the Dell today he'll say 'while I was
over on the laptop, this is what happened'").

No real ssh/scp is run and no live memory store is touched: the runners and
the session / fact sinks are fakes, files go to a temp dir.

    python -m unittest tests.test_peer_memory
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import types
import unittest
from unittest import mock

from core import peer_memory as M

NOW = 1_800_000_000.0


def _ran(stdout="", returncode=0):
    return types.SimpleNamespace(stdout=stdout, returncode=returncode)


class BundleTests(unittest.TestCase):
    def test_only_this_machines_own_recent_memory_goes_out(self):
        sessions = [
            {"ts": NOW - 3600, "iso_start": "a", "summary": "Opened Accelo."},
            {"ts": NOW - 3600, "iso_start": "b", "summary": "On the desktop: x",
             "from_peer": "desktop"},                          # imported: no echo
            {"ts": NOW - 40 * 86400, "iso_start": "c", "summary": "old"},
            {"ts": NOW, "iso_start": "d", "summary": ""},
        ]
        facts = [
            {"text": "the owner's Teams is the new MSIX app", "created_at": NOW - 60, "tags": ["pc"]},
            {"text": "from the laptop", "created_at": NOW - 60, "source": "peer:laptop"},
            {"text": "ancient", "created_at": NOW - 90 * 86400},
        ]
        b = M.build_bundle(sessions, facts, now=NOW, host="DESK")
        self.assertEqual(b["host"], "DESK")
        self.assertEqual([s["iso_start"] for s in b["sessions"]], ["a"])
        self.assertEqual([f["text"] for f in b["facts"]], ["the owner's Teams is the new MSIX app"])
        self.assertEqual(b["facts"][0]["tags"], ["pc"])


class PeerEntryTests(unittest.TestCase):
    def test_peer_sessions_say_where_they_happened_once(self):
        b = {"sessions": [{"iso_start": "2026-10-19T09:00:00", "ts": NOW, "summary": "Opened Teams."},
                          {"iso_start": "x", "summary": "On the Dell laptop: already"},
                          {"iso_start": "y", "summary": ""}]}
        out = M.peer_session_entries(b, "Dell laptop")
        self.assertEqual([e["summary"] for e in out],
                         ["On the Dell laptop: Opened Teams.", "On the Dell laptop: already"])
        self.assertTrue(all(e["from_peer"] == "Dell laptop" and e["location"] == "Dell laptop"
                            for e in out))


class ImportTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.path = os.path.join(self.dir, "peer_memory_from_DELL.json")

    def _bundle(self, written_at, facts=("the guest likes the fan on",)):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump({"host": "DELL", "written_at": written_at,
                       "sessions": [{"iso_start": "s1", "ts": NOW, "summary": "Did homework."}],
                       "facts": [{"text": t, "tags": []} for t in facts]}, f)

    def test_each_version_of_a_bundle_is_imported_once(self):
        merged, added, marks = [], [], {}
        merge = lambda entries: merged.extend(entries) or len(entries)
        add = lambda text, **kw: added.append((text, kw["source"]))
        self._bundle(NOW)
        self.assertEqual(M.import_bundle(self.path, "Dell laptop", merge, add, marks), (1, 1))
        self.assertEqual(merged[0]["summary"], "On the Dell laptop: Did homework.")
        self.assertEqual(added, [("the guest likes the fan on", "peer:Dell laptop")])
        self.assertEqual(marks, {"DELL": NOW})
        # Same version again: nothing.
        self.assertEqual(M.import_bundle(self.path, "Dell laptop", merge, add, marks), (0, 0))
        # A newer version: imported.
        self._bundle(NOW + 600)
        self.assertEqual(M.import_bundle(self.path, "Dell laptop", merge, add, marks), (1, 1))

    def test_a_broken_bundle_or_failing_fact_never_raises(self):
        with open(self.path, "w") as f:
            f.write("{not json")
        self.assertEqual(M.import_bundle(self.path, "x", lambda e: 0, lambda t, **k: None, {}), (0, 0))
        self._bundle(NOW, facts=("a", "b"))
        def add(text, **kw):
            if text == "a":
                raise RuntimeError("store busy")
        self.assertEqual(M.import_bundle(self.path, "x", lambda e: len(e), add, {}), (1, 1))


class TransferTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, True)

    def test_push_copies_the_bundle_under_this_hosts_name(self):
        src = os.path.join(self.dir, "self.json")
        open(src, "w").write("{}")
        run = mock.Mock(return_value=_ran())
        with mock.patch.object(M, "_host", return_value="DESK"):
            self.assertTrue(M.push("brade@peer-host", run=run, path=src))
        argv = run.call_args[0][0]
        self.assertEqual(argv[:3], ["scp", "-B", "-q"])
        self.assertTrue(argv[-1].startswith("brade@peer-host:"))
        self.assertTrue(argv[-1].endswith("/peer_memory_from_DESK.json"))
        self.assertFalse(M.push("", run=run, path=src))
        self.assertFalse(M.push("brade@peer-host", run=mock.Mock(return_value=_ran("", 1)), path=src))

    def test_pull_saves_the_peers_bundle_by_its_host(self):
        bundle = {"host": "DELL", "written_at": NOW, "sessions": [], "facts": []}
        run = mock.Mock(return_value=_ran(json.dumps(bundle)))
        with mock.patch.object(M, "_DATA", self.dir):
            dest = M.pull("brade@peer-host", run=run)
        self.assertEqual(os.path.basename(dest), "peer_memory_from_DELL.json")
        with open(dest, encoding="utf-8") as f:
            self.assertEqual(json.load(f)["host"], "DELL")

    def test_pull_from_an_offline_peer_is_none(self):
        self.assertIsNone(M.pull("brade@peer-host", run=mock.Mock(return_value=_ran("", 255))))
        self.assertIsNone(M.pull("brade@peer-host", run=mock.Mock(return_value=_ran("garbage"))))
        self.assertIsNone(M.pull(""))

    def test_the_self_bundle_is_ascii_for_the_remote_console(self):
        path = os.path.join(self.dir, "self.json")
        self.assertTrue(M.write_self_bundle({"summary": "The owner’s fan — on"}, path=path))
        raw = open(path, "rb").read()
        self.assertTrue(all(b < 128 for b in raw))
        self.assertEqual(json.loads(raw)["summary"], "The owner’s fan — on")


class SwitchTests(unittest.TestCase):
    def test_off_without_a_peer(self):
        with mock.patch.object(M._handoff, "enabled", return_value=False), \
                mock.patch.object(M.threading, "Thread") as thread:
            self.assertFalse(M.start(log=lambda m: None))
        thread.assert_not_called()


if __name__ == "__main__":
    unittest.main()
