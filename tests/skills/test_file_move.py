"""skills.file_move - move_file / undo_last_move (v3 W15, 2026-10-27).

EVERY file operation here happens inside a fresh temp directory: a fake
user profile (Downloads, Documents, ...) and a fake Nextcloud folder are
built under tempfile.mkdtemp() and the skill's environment seams
(home_dir / known_folders / nextcloud_root / refused_prefixes / journal_path)
point at them. Real refused paths (C:\\Windows, C:\\JARVIS ...) are only ever
CLASSIFIED, never touched.

    python -m unittest tests.skills.test_file_move
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

from skills import file_move as FM

T = FM.TERMINAL


class FakeProfile:
    """A throwaway profile + Nextcloud tree, wired into the skill's seams."""

    FOLDERS = ("Downloads", "Desktop", "Documents", "Pictures", "Music", "Videos")

    def __init__(self, testcase: unittest.TestCase):
        self.root = os.path.realpath(tempfile.mkdtemp(prefix="jarvis_fm_"))
        testcase.addCleanup(shutil.rmtree, self.root, True)
        self.home = os.path.join(self.root, "home")
        self.nc = os.path.join(self.root, "Nextcloud")
        self.data = os.path.join(self.root, "data")
        self.sys = os.path.join(self.root, "Windows")
        for d in (self.home, self.nc, self.data, self.sys):
            os.makedirs(d)
        self.known = {}
        for name in self.FOLDERS:
            p = os.path.join(self.home, name)
            os.makedirs(p)
            self.known[name.lower()] = p
        os.makedirs(os.path.join(self.nc, "Documents"))
        os.makedirs(os.path.join(self.nc, "3D Printing"))
        for attr, value in (
                ("home_dir", lambda: self.home),
                ("known_folders", lambda: dict(self.known)),
                ("nextcloud_root", lambda: self.nc),
                ("refused_prefixes", lambda: [self.sys, self.data]),
                ("journal_path", lambda: os.path.join(self.data, "file_moves.json"))):
            p = mock.patch.object(FM, attr, value)
            p.start()
            testcase.addCleanup(p.stop)
        p = mock.patch.object(FM, "_plans", {})
        p.start()
        testcase.addCleanup(p.stop)
        p = mock.patch.object(FM, "_failures", {})
        p.start()
        testcase.addCleanup(p.stop)

    def file(self, *parts, data=b"hello"):
        path = os.path.join(self.home, *parts) if not os.path.isabs(parts[0]) \
            else os.path.join(*parts)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)
        return path

    def dl(self, name, data=b"hello"):
        return self.file("Downloads", name, data=data)

    def journal(self):
        with open(os.path.join(self.data, "file_moves.json"), encoding="utf-8") as f:
            return json.load(f)["moves"]


def _quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def ask_and_yes(name, arg):
    """The confirm path in miniature: the dry run, then the action."""
    q = _quiet(FM.confirm_question, name, arg)
    res = _quiet(FM.move_file if name == "move_file" else FM.undo_last_move, arg)
    return q, res


class ParseTests(unittest.TestCase):
    def test_what_and_where(self):
        self.assertEqual(FM.parse_arg("my resume to documents")[:3],
                         ("my resume", "", "documents"))
        self.assertEqual(FM.parse_arg("Move the invoice from downloads to the desktop.")[:3],
                         ("the invoice", "downloads", "the desktop"))
        self.assertEqual(FM.parse_arg("notes to self.txt into nextcloud")[:3],
                         ("notes to self.txt", "", "nextcloud"))

    def test_no_destination(self):
        self.assertEqual(FM.parse_arg("my resume"), ("", "", "", ""))


class SourceResolutionTests(unittest.TestCase):
    def setUp(self):
        self.p = FakeProfile(self)

    def test_found_by_name_words(self):
        src = self.p.dl("Sample CV Resume.pdf")
        self.assertEqual(FM.resolve_source("my resume"), (src, ""))

    def test_explicit_path(self):
        src = self.p.file("Desktop", "a.txt")
        self.assertEqual(FM.resolve_source(src), (src, ""))

    def test_more_than_one_match_asks_which_never_guesses(self):
        for n in ("Resume.pdf", "Resume old.docx", "resume-2025.txt", "my resume.md"):
            self.p.dl(n)
        src, why = FM.resolve_source("resume")
        self.assertIsNone(src)
        self.assertTrue(why.startswith("I found 4 files; the first three are 'Resume.pdf' in Downloads"), why)
        self.assertEqual(why.count("' in Downloads"), 3)
        self.assertTrue(why.endswith("Which one, sir?"))

    def test_from_hint_limits_the_search(self):
        self.p.dl("report.pdf")
        doc = self.p.file("Documents", "report.pdf")
        self.assertEqual(FM.resolve_source("report", "documents"), (doc, ""))

    def test_name_holding_in_is_found_whole(self):
        src = self.p.dl("notes in progress.txt")
        plan, why = FM.plan_move("notes in progress to documents")
        self.assertEqual(plan["src"], src, why)

    def test_pictures_inside_nextcloud_is_one_match_not_two(self):
        pics = os.path.join(self.p.nc, "Pictures")
        os.makedirs(pics)
        self.p.known["pictures"] = pics
        src = self.p.file(pics, "holiday.jpg")
        self.assertEqual(FM.resolve_source("holiday"), (src, ""))

    def test_word_start_only(self):
        self.p.dl("precv.txt")
        src, why = FM.resolve_source("cv")
        self.assertIsNone(src)
        self.assertIn("couldn't find a file called 'cv'", why)

    def test_folder_is_refused_files_only(self):
        os.makedirs(os.path.join(self.p.home, "Downloads", "Taxes"))
        self.assertEqual(FM.resolve_source("taxes")[1],
                         "I can only move single files for now, sir.")
        self.assertEqual(FM.resolve_source(os.path.join(self.p.home, "Downloads", "Taxes"))[1],
                         "I can only move single files for now, sir.")

    def test_hidden_and_icloud_files_never_found(self):
        self.p.dl(".resume.pdf")
        self.p.file("iCloudDrive", "resume.pdf")
        src, why = FM.resolve_source("resume")
        self.assertIsNone(src)
        self.assertIn("couldn't find", why)

    def test_index_hits_used_only_when_name_matches(self):
        deep = self.p.file("Documents", "a", "b", "c", "d", "e", "Resume final.pdf")
        other = self.p.file("Documents", "a", "b", "c", "d", "e", "notes.txt")

        class Rag:
            def is_available(self):
                return True

            def search(self, q, k=5):
                return [{"path": other}, {"path": deep}, {"path": r"C:\Windows\resume.pdf"}]

        mod = mock.Mock()
        mod._rag.return_value = Rag()
        with mock.patch.dict("sys.modules", {"skill_personal_rag": mod}):
            self.assertEqual(FM.resolve_source("resume final"), (deep, ""))


class DestinationResolutionTests(unittest.TestCase):
    def setUp(self):
        self.p = FakeProfile(self)

    def test_folder_words(self):
        for word, key in (("documents", "documents"), ("my documents folder", "documents"),
                          ("the desktop", "desktop"), ("downloads", "downloads"),
                          ("photos", "pictures"), ("music", "music"), ("videos", "videos")):
            with self.subTest(word=word):
                self.assertEqual(FM.resolve_folder(word), (self.p.known[key], ""))

    def test_nextcloud_folders(self):
        self.assertEqual(FM.resolve_folder("nextcloud"), (self.p.nc, ""))
        self.assertEqual(FM.resolve_folder("nextcloud documents"),
                         (os.path.join(self.p.nc, "Documents"), ""))
        self.assertEqual(FM.resolve_folder("3d printing in nextcloud"),
                         (os.path.join(self.p.nc, "3D Printing"), ""))

    def test_folder_name_and_explicit_path(self):
        self.assertEqual(FM.resolve_folder("3D printing folder"),
                         (os.path.join(self.p.nc, "3D Printing"), ""))
        target = os.path.join(self.p.home, "Documents")
        self.assertEqual(FM.resolve_folder(target), (target, ""))

    def test_missing_folder(self):
        self.assertIsNone(FM.resolve_folder("taxes")[0])
        self.assertIsNone(FM.resolve_folder(os.path.join(self.p.home, "nope"))[0])

    def test_two_folders_of_that_name_ask_which(self):
        os.makedirs(os.path.join(self.p.home, "Documents", "School"))
        os.makedirs(os.path.join(self.p.nc, "School"))
        dest, why = FM.resolve_folder("school")
        self.assertIsNone(dest)
        self.assertTrue(why.endswith("Which one, sir?"), why)
        self.assertIn("School in Documents", why)
        self.assertIn("School in Nextcloud", why)


class RefusalTests(unittest.TestCase):
    """Every refused root - classified only, never touched."""

    def test_real_system_and_jarvis_roots(self):
        for p in (r"C:\Windows\System32\notepad.exe", r"C:\Program Files\x\a.exe",
                  r"C:\Program Files (x86)\x\a.exe", r"C:\ProgramData\x\a.txt",
                  r"C:\JARVIS\data\memory.json", r"C:\JARVIS\bobert_companion.py"):
            with self.subTest(p=p):
                self.assertIn("system or JARVIS folder", FM.refusal(p))

    def test_this_checkout_and_its_data_dir_are_refused(self):
        here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.assertIn("system or JARVIS folder",
                      FM.refusal(os.path.join(here, "skills", "file_move.py")))
        from core import paths
        self.assertIn("system or JARVIS folder",
                      FM.refusal(os.path.join(paths.data_dir(create=False), "x.json")))

    def test_drive_roots(self):
        for p in ("C:\\", "D:\\"):
            with self.subTest(p=p):
                self.assertIn("drive's top folder", FM.refusal(p))


class RefusalInProfileTests(unittest.TestCase):
    def setUp(self):
        self.p = FakeProfile(self)

    def test_outside_allowed_roots(self):
        outside = os.path.join(self.p.root, "elsewhere", "a.txt")
        self.assertIn("outside your own folders", FM.refusal(outside))

    def test_appdata(self):
        f = self.p.file("AppData", "Roaming", "app", "a.txt")
        self.assertIn("AppData", FM.refusal(f))
        self.assertIn("AppData", FM.resolve_source(f)[1])

    def test_icloud_and_apple_folders(self):
        for parts in (("iCloudDrive", "a.txt"), ("iCloud Photos", "Photos", "b.jpg"),
                      ("Apple", "MobileSync", "Backup", "c.plist")):
            with self.subTest(parts=parts):
                f = self.p.file(*parts)
                self.assertIn("never move Apple data", FM.refusal(f))
        d = os.path.join(self.p.home, "iCloudDrive")
        self.assertIn("never move Apple data", FM.resolve_folder(d)[1])
        self.assertIsNone(FM.resolve_folder("iclouddrive")[0])

    def test_a_file_merely_named_apple_is_fine(self):
        f = self.p.dl("apple pie.docx")
        self.assertEqual(FM.refusal(f), "")
        self.assertEqual(FM.resolve_source("apple pie"), (f, ""))

    def test_hidden_and_dot_files(self):
        dot = self.p.dl(".secret.txt")
        self.assertIn("hidden or system", FM.refusal(dot))
        f = self.p.dl("plain.txt")
        if os.name == "nt":
            import ctypes
            ctypes.windll.kernel32.SetFileAttributesW(f, FM.FILE_ATTRIBUTE_HIDDEN)
            self.addCleanup(ctypes.windll.kernel32.SetFileAttributesW, f, 0x80)
            self.assertIn("hidden or system", FM.refusal(f))

    @unittest.skipUnless(os.name == "nt", "junctions are Windows-only")
    def test_junction_on_the_way(self):
        import _winapi
        target = os.path.join(self.p.nc, "Documents")
        link = os.path.join(self.p.home, "Linked")
        try:
            _winapi.CreateJunction(target, link)
        except Exception as e:   # pragma: no cover
            self.skipTest(f"cannot create a junction here: {e}")
        f = self.p.file(target, "x.txt")
        via = os.path.join(link, "x.txt")
        self.assertTrue(os.path.isfile(via))
        self.assertIn("junction", FM.refusal(via))
        self.assertIn("junction", FM.refusal(link))
        self.assertEqual(FM.refusal(f), "")


class MoveTests(unittest.TestCase):
    def setUp(self):
        self.p = FakeProfile(self)

    def test_question_names_file_and_both_folders(self):
        self.p.dl("Sample CV Resume.pdf")
        q, _why = _quiet(FM.confirm_question, "move_file", "my resume to documents")
        self.assertEqual(q, "Move 'Sample CV Resume.pdf' from Downloads to Documents, sir?")

    def test_dry_run_touches_nothing(self):
        src = self.p.dl("resume.pdf")
        _quiet(FM.confirm_question, "move_file", "resume to documents")
        self.assertTrue(os.path.isfile(src))
        self.assertEqual(os.listdir(self.p.known["documents"]), [])

    def test_yes_moves_and_verifies(self):
        src = self.p.dl("resume.pdf", data=b"x" * 1234)
        q, res = ask_and_yes("move_file", "resume to documents")
        dest = os.path.join(self.p.known["documents"], "resume.pdf")
        self.assertEqual(res, "Done, sir - 'resume.pdf' is now in Documents.")
        self.assertFalse(os.path.exists(src))
        self.assertEqual(os.path.getsize(dest), 1234)

    def test_without_the_question_nothing_moves(self):
        src = self.p.dl("resume.pdf")
        res = _quiet(FM.move_file, "resume to documents")
        self.assertTrue(res.startswith(T), res)
        self.assertIn("after you've said yes", res)
        self.assertTrue(os.path.isfile(src))
        self.assertEqual(os.listdir(self.p.known["documents"]), [])

    def test_a_stale_plan_moves_nothing(self):
        src = self.p.dl("resume.pdf")
        _quiet(FM.confirm_question, "move_file", "resume to documents")
        for plan in FM._plans.values():
            plan["at"] -= FM.PLAN_TTL_S + 1
        res = _quiet(FM.move_file, "resume to documents")
        self.assertIn("after you've said yes", res)
        self.assertTrue(os.path.isfile(src))

    def test_never_overwrites_proposes_numbered_name(self):
        self.p.dl("resume.pdf", data=b"new!")
        keep = self.p.file("Documents", "resume.pdf", data=b"old")
        # Two files share the name now, so the source is named by its folder.
        q, res = ask_and_yes("move_file", "resume from downloads to documents")
        self.assertEqual(q[0], "Move 'resume.pdf' from Downloads to Documents, sir? "
                               "There's already one there, so I'll call it 'resume (1).pdf'.")
        self.assertEqual(res, "Done, sir - 'resume.pdf' is now in Documents, as 'resume (1).pdf'.")
        with open(keep, "rb") as f:
            self.assertEqual(f.read(), b"old")
        with open(os.path.join(self.p.known["documents"], "resume (1).pdf"), "rb") as f:
            self.assertEqual(f.read(), b"new!")

    def test_name_taken_after_the_question_moves_nothing(self):
        src = self.p.dl("resume.pdf")
        _quiet(FM.confirm_question, "move_file", "resume to documents")
        self.p.file("Documents", "resume.pdf", data=b"arrived")
        res = _quiet(FM.move_file, "resume to documents")
        self.assertTrue(res.startswith(T))
        self.assertIn("already in Documents", res)
        self.assertTrue(os.path.isfile(src))

    def test_file_changed_after_the_question_moves_nothing(self):
        src = self.p.dl("resume.pdf")
        _quiet(FM.confirm_question, "move_file", "resume to documents")
        with open(src, "ab") as f:
            f.write(b"more")
        res = _quiet(FM.move_file, "resume to documents")
        self.assertIn("changed since I asked", res)
        self.assertTrue(os.path.isfile(src))

    def test_already_there(self):
        self.p.file("Documents", "resume.pdf")
        self.assertIsNone(_quiet(FM.confirm_question, "move_file", "resume to documents"))
        res = _quiet(FM.move_file, "resume to documents")
        self.assertEqual(res, T + "'resume.pdf' is already in Documents, sir.")

    def test_not_found_is_terminal_and_which_one_is_a_question(self):
        self.assertEqual(_quiet(FM.move_file, "taxes to documents"),
                         T + "I couldn't find a file called 'taxes' in your folders, sir.")
        self.p.dl("a resume.pdf")
        self.p.dl("b resume.pdf")
        res = _quiet(FM.move_file, "resume to documents")
        self.assertFalse(res.startswith(T))
        self.assertTrue(res.endswith("Which one, sir?"))

    def test_into_a_refused_folder(self):
        self.p.dl("resume.pdf")
        os.makedirs(os.path.join(self.p.home, "iCloudDrive"))
        dest = os.path.join(self.p.home, "iCloudDrive")
        self.assertIsNone(_quiet(FM.confirm_question, "move_file", f"resume to {dest}"))
        self.assertIn("never move Apple data", _quiet(FM.move_file, f"resume to {dest}"))
        self.assertIn("system or JARVIS folder",
                      _quiet(FM.move_file, f"resume to {self.p.sys}"))

    @unittest.skipUnless(os.name == "nt", "Windows sharing rules")
    def test_open_file_fails_and_says_so(self):
        src = self.p.dl("resume.pdf")
        _quiet(FM.confirm_question, "move_file", "resume to documents")
        with open(src, "rb"):
            res = _quiet(FM.move_file, "resume to documents")
        self.assertTrue(res.startswith(T), res)
        self.assertIn("open or locked by another program", res)
        self.assertTrue(os.path.isfile(src))
        self.assertEqual(os.listdir(self.p.known["documents"]), [])

    def test_cross_drive_path_copies_checks_then_removes(self):
        src = self.p.dl("resume.pdf", data=b"y" * 999)
        with mock.patch.object(FM, "_same_drive", return_value=False):
            q, res = ask_and_yes("move_file", "resume to nextcloud documents")
        self.assertEqual(res, "Done, sir - 'resume.pdf' is now in Documents.")
        dest = os.path.join(self.p.nc, "Documents", "resume.pdf")
        self.assertEqual(os.path.getsize(dest), 999)
        self.assertFalse(os.path.exists(src))
        self.assertEqual(os.listdir(os.path.join(self.p.nc, "Documents")), ["resume.pdf"])

    def test_cross_drive_bad_copy_keeps_the_original(self):
        src = self.p.dl("resume.pdf", data=b"z" * 50)

        def short_copy(a, b):
            with open(b, "wb") as f:
                f.write(b"z")

        with mock.patch.object(FM, "_same_drive", return_value=False), \
                mock.patch.object(FM.shutil, "copy2", side_effect=short_copy):
            _q, res = ask_and_yes("move_file", "resume to nextcloud documents")
        self.assertTrue(res.startswith(T))
        self.assertIn("wrong size", res)
        self.assertTrue(os.path.isfile(src))
        self.assertEqual(os.listdir(os.path.join(self.p.nc, "Documents")), [])

    def test_success_only_after_verify(self):
        self.p.dl("resume.pdf")
        with mock.patch.object(FM, "verify", return_value="'resume.pdf' is the wrong size"):
            _q, res = ask_and_yes("move_file", "resume to documents")
        self.assertEqual(res, T + "The move didn't check out, sir: 'resume.pdf' is the wrong size.")

    def test_verify_itself(self):
        a = self.p.dl("a.txt", data=b"12345")
        b = os.path.join(self.p.known["documents"], "a.txt")
        self.assertIn("can't see", FM.verify(a, b, 5))
        shutil.copy2(a, b)
        self.assertIn("still in Downloads", FM.verify(a, b, 5))
        self.assertIn("wrong size", FM.verify(a, b, 6))
        os.remove(a)
        self.assertEqual(FM.verify(a, b, 5), "")


class JournalAndUndoTests(unittest.TestCase):
    def setUp(self):
        self.p = FakeProfile(self)

    def test_journal_written_atomically(self):
        self.p.dl("resume.pdf")
        from core import atomic_io
        with mock.patch.object(atomic_io, "_atomic_write_json",
                               wraps=atomic_io._atomic_write_json) as w:
            ask_and_yes("move_file", "resume to documents")
        w.assert_called_once()
        self.assertEqual(w.call_args[0][0], os.path.join(self.p.data, "file_moves.json"))
        (entry,) = self.p.journal()
        self.assertEqual(entry["kind"], "move")
        self.assertEqual(entry["dest"], os.path.join(self.p.known["documents"], "resume.pdf"))
        self.assertFalse(entry["undone"])

    def test_undo_moves_it_back_after_its_own_question(self):
        src = self.p.dl("resume.pdf")
        ask_and_yes("move_file", "resume to documents")
        self.assertIn("after you've said yes", _quiet(FM.undo_last_move, ""))
        q, res = ask_and_yes("undo_last_move", "")
        self.assertEqual(q[0], "Move 'resume.pdf' back from Documents to Downloads, sir?")
        self.assertEqual(res, "Done, sir - 'resume.pdf' is back in Downloads.")
        self.assertTrue(os.path.isfile(src))
        self.assertEqual(os.listdir(self.p.known["documents"]), [])
        moves = self.p.journal()
        self.assertTrue(moves[0]["undone"])
        self.assertEqual(_quiet(FM.undo_last_move, ""),
                         T + "I haven't moved any files to undo, sir.")

    def test_undo_refuses_when_the_file_moved_on(self):
        self.p.dl("resume.pdf")
        ask_and_yes("move_file", "resume to documents")
        os.rename(os.path.join(self.p.known["documents"], "resume.pdf"),
                  os.path.join(self.p.known["desktop"], "resume.pdf"))
        self.assertIsNone(_quiet(FM.confirm_question, "undo_last_move", ""))
        res = _quiet(FM.undo_last_move, "")
        self.assertTrue(res.startswith(T))
        self.assertIn("isn't where I put it", res)

    def test_undo_never_overwrites_either(self):
        self.p.dl("resume.pdf", data=b"moved")
        ask_and_yes("move_file", "resume to documents")
        self.p.dl("resume.pdf", data=b"newer")
        q, res = ask_and_yes("undo_last_move", "")
        self.assertIn("'resume (1).pdf'", q[0])
        with open(os.path.join(self.p.known["downloads"], "resume.pdf"), "rb") as f:
            self.assertEqual(f.read(), b"newer")

    def test_no_journal_means_nothing_to_undo(self):
        self.assertIsNone(_quiet(FM.confirm_question, "undo_last_move", ""))


class RegistrationTests(unittest.TestCase):
    def test_registers_both_and_declares_verbatim(self):
        actions = {}
        with mock.patch.object(FM, "_cfg", return_value=True):
            FM.register(actions)
        self.assertEqual(set(actions), {"move_file", "undo_last_move"})
        self.assertEqual(set(FM.SPEAK_VERBATIM_ACTIONS), {"move_file", "undo_last_move"})
        self.assertIn("[ACTION: move_file, resume to documents]", FM.PROMPT_EXAMPLES)
        self.assertIn("[ACTION: undo_last_move]", FM.PROMPT_EXAMPLES)

    def test_setting_off_registers_nothing(self):
        actions = {}
        with mock.patch.object(FM, "_cfg", return_value=False):
            _quiet(FM.register, actions)
        self.assertEqual(actions, {})

    def test_setting_declared_on(self):
        from core import config
        self.assertIs(config.FILE_MOVE_ENABLED, True)

    def test_terminal_prefix_matches_core(self):
        from core.failure_markers import TERMINAL_FAILURE_PREFIX
        self.assertEqual(FM.TERMINAL, TERMINAL_FAILURE_PREFIX)


if __name__ == "__main__":
    unittest.main()
