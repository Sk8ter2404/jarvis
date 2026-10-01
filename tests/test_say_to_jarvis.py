"""tools/say_to_jarvis.py tags its injects as test input (B023, 2026-10-01).

Every inject used to count as the owner typing, and a typed turn always
teaches (core/learn_gate.py), so a tester's line ("my favourite colour is
teal", fed in to check a reply) was learned as a fact about the owner. The
tool now writes ``"source": "test"``, which bobert_companion reads into
_last_inject_source and the main loop never learns from.

The queue path is redirected into a temp dir: the live injected_commands.json
is never touched.

    python -m unittest tests.test_say_to_jarvis
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

_HERE = os.path.dirname(os.path.abspath(__file__))
_TOOL = os.path.join(os.path.dirname(_HERE), "tools", "say_to_jarvis.py")


def _load_tool():
    spec = importlib.util.spec_from_file_location("_say_to_jarvis_under_test",
                                                  _TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class EnqueueSourceTests(unittest.TestCase):
    def setUp(self):
        self.tool = _load_tool()
        self.tmp = tempfile.mkdtemp(prefix="say_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.queue = os.path.join(self.tmp, "injected_commands.json")
        for name, value in (("PROJECT_DIR", self.tmp),
                            ("QUEUE_PATH", self.queue)):
            p = mock.patch.object(self.tool, name, value)
            p.start()
            self.addCleanup(p.stop)

    def _items(self):
        with open(self.queue, encoding="utf-8") as f:
            return json.load(f)

    def test_each_line_is_tagged_as_test_input(self):
        self.tool.enqueue("what time is it")
        self.tool.enqueue("my favourite colour is teal")
        items = self._items()
        self.assertEqual([i["text"] for i in items],
                         ["what time is it", "my favourite colour is teal"])
        self.assertEqual({i.get("source") for i in items}, {"test"})

    def test_an_owner_line_already_queued_keeps_its_own_shape(self):
        with open(self.queue, "w", encoding="utf-8") as f:
            json.dump([{"text": "typed on the web page", "ts": 1.0}], f)
        self.tool.enqueue("a tester line")
        items = self._items()
        self.assertNotIn("source", items[0])
        self.assertEqual(items[1].get("source"), "test")


if __name__ == "__main__":
    unittest.main()
