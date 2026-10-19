"""skills.backblaze_status - READ-ONLY Backblaze progress (2026-10-19). Reads
fixture copies of the client's own report files from a temp dir.

    python -m unittest tests.skills.test_backblaze_status
"""
from __future__ import annotations

import os
import shutil
import tempfile
import unittest

from skills import backblaze_status as B

TOTAL = '<?xml version="1.0"?>\n<contents>\n<totals totnumfilesforbackup="2120420" totnumbytesforbackup="5589208620721" />\n</contents>\n'
LEFT = '<?xml version="1.0"?>\n<contents>\n<remaining remainingnumfilesforbackup="1104" remainingnumbytesforbackup="33079603706" />\n</contents>\n'
LAST = '<?xml version="1.0"?>\n<contents>\n<lastbackupcompleted gmt_millis="1792432205484" localdatetime="20261019125005" />\n</contents>\n'


class BackblazeTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, True)

    def _write(self, **files):
        for name, text in files.items():
            with open(os.path.join(self.dir, name), "w", encoding="utf-8") as f:
                f.write(text)

    def test_live_numbers_read_back(self):
        self._write(**{"bzstat_totalbackup.xml": TOTAL, "bzstat_remainingbackup.xml": LEFT,
                       "bzstat_lastbackupcompleted.xml": LAST})
        st = B.read_status(self.dir)
        self.assertEqual((st["total_bytes"], st["left_files"], st["left_bytes"]),
                         (5589208620721, 1104, 33079603706))
        line = B.describe(st, now=st["last_done"] + 60)
        self.assertTrue(line.startswith("Backblaze has 99.4 percent of your 5.59 TB backed up, sir, "
                                        "with 1,104 files, 33.1 GB still to go."), line)
        self.assertIn("The last full pass finished", line)
        self.assertIn("today", line)

    def test_nothing_left_and_missing_files(self):
        self._write(**{"bzstat_totalbackup.xml": TOTAL,
                       "bzstat_remainingbackup.xml": LEFT.replace('"1104"', '"0"').replace('"33079603706"', '"0"')})
        self.assertIn("100.0 percent", B.describe(B.read_status(self.dir)))
        self.assertIn("nothing waiting", B.describe(B.read_status(self.dir)))
        empty = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, empty, True)
        self.assertEqual(B.describe(B.read_status(empty)),
                         "I couldn't read Backblaze's status files just now, sir.")

    def test_registered(self):
        actions = {}
        B.register(actions)
        for a in ("backblaze_status", "backup_status", "how_is_the_backup"):
            self.assertIn(a, actions)


if __name__ == "__main__":
    unittest.main()
