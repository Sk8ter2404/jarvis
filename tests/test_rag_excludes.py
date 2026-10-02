"""File search must never index password / secret files.

core/config.py ships RAG_EXCLUDE_GLOBS: the indexer's structural excludes
(.git, node_modules, AppData...) PLUS secret-shaped names (password, token,
private key, .env, .pem, .kdbx, wallet, recovery and backup codes, BitLocker,
2FA, device exports) and every .csv / .tsv export. skills/personal_rag.py
pushes it into core.rag_indexer.configure() with the other RAG_* knobs.

What these tests pin:
  * the excluded / included name tables, on Windows and POSIX paths, either
    slash, any case;
  * a pattern without a slash is tested against the file NAME and every folder
    name BELOW the watched root, never against the folders above it (a user
    folder called "Compass" must not hide everything);
  * an owner override saved in user_settings.json REPLACES the list;
  * a file that is already indexed and later matches an exclude is DROPPED
    from the index on the next scan (it used to stay searchable forever,
    because the garbage-collect pass only removed files missing from disk);
  * search refuses an excluded file's hits at once, before that scan's
    clean-up pass has run.

All names here are made up. stdlib unittest + unittest.mock only; no chromadb,
no Ollama, no real folders outside a tempdir.
"""
from __future__ import annotations

import ast
import os
import tempfile
import unittest
from unittest import mock

from core import config
from core import rag_indexer as rag

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _shipped_excludes() -> list:
    """The RAG_EXCLUDE_GLOBS literal as SHIPPED in core/config.py.

    Read from the source, not from the imported module, because
    _apply_user_settings() may have replaced the live value from a settings
    file on the box running the tests."""
    with open(os.path.join(_ROOT, "core", "config.py"), encoding="utf-8") as f:
        tree = ast.parse(f.read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "RAG_EXCLUDE_GLOBS"
                for t in node.targets):
            return list(ast.literal_eval(node.value))
    raise AssertionError("core/config.py defines no RAG_EXCLUDE_GLOBS")


# The indexer's own defaults before the secret patterns were added. Every one
# must survive in the shipped config list ("current defaults PLUS ...").
_ORIGINAL_INDEXER_DEFAULTS = (
    "*/.git/*", "*/node_modules/*", "*/__pycache__/*", "*/.venv/*",
    "*/venv/*", "*/dist/*", "*/build/*", "*/.cache/*", "*/.next/*",
    "*/Library/Caches/*", "*/AppData/Local/*", "*/AppData/Roaming/*",
    "*.tmp", "*.lock", "*.cache",
)

# File names that must NEVER be indexed (all made up).
_EXCLUDED_NAMES = (
    "passwords.txt", "Client PASSWORD list.md", "router passwd.txt",
    "pass.txt", "Passphrase.md",
    "my_secret_notes.md", "client-secrets.yaml",
    "AWS-Credentials.txt", "creds.json",
    "github_token.txt", "Tokens.md",
    "ApiKey.txt", "service_api_key.json", "api-key notes.md",
    "Private Key.txt", "private_key.txt", "privatekey.md",
    ".env", ".env.local", "prod.env",
    "id_rsa", "id_rsa.pub",
    "server.pem", "server.pem.txt", "tls.key", "tls.key.txt",
    "cert.pfx", "vault.kdbx", "vault.kdbx.bak",
    "crypto wallet seed.txt", "Wallet.json",
    "Recovery Codes.txt", "account-recovery.md",
    "BitLocker Recovery Key 0A1B2C3D.TXT", "bitlocker.txt",
    "2FA backup.txt", "my 2fa.md",
    "mail backup codes.txt", "backup_codes.md", "BackupCodes.md",
    "level-devices.csv", "Level_Devices.md", "level devices export.txt",
    "inventory.csv", "Devices.TSV", "export.CSV",
)

# Near misses that must STILL be indexed.
_INCLUDED_NAMES = (
    "meeting notes.md", "budget 2026.md", "README.md", "lab report.docx",
    "keyboard shortcuts.md", "monkey notes.txt", "keynote outline.md",
    "tokyo trip.md", "credit union notes.md", "environment setup.md",
    "walleye fishing.md", "discovery channel.md", "level design.md",
    "devices.md", "backup plan.md", "code review.md",
)

_WIN_ROOT = r"C:\Users\Someone\Documents"
_POSIX_ROOT = "/home/someone/docs"


def _spellings(root: str, rel: str):
    """One relative path under `root`, spelled every way Windows hands it
    over: backslashes, forward slashes, and with the case flipped."""
    back = root.replace("/", "\\").rstrip("\\") + "\\" + rel.replace("/", "\\")
    fwd = root.replace("\\", "/").rstrip("/") + "/" + rel.replace("\\", "/")
    return (back, fwd, back.upper(), fwd.lower())


class _ExcludeBase(unittest.TestCase):
    """Snapshot + restore the indexer globals each test touches."""

    def setUp(self):
        saved = {k: getattr(rag, k) for k in (
            "RAG_EXCLUDE_GLOBS", "RAG_INDEX_PATHS", "_collection",
            "_embed_model", "_observer", "_last_error",
            "_last_full_scan_ts")}
        stats = dict(rag._stats)

        def _restore():
            for k, v in saved.items():
                setattr(rag, k, v)
            rag._stats.clear()
            rag._stats.update(stats)
            rag._stop_flag.clear()
            self._drain_queue()
        self.addCleanup(_restore)
        pp = mock.patch("builtins.print", lambda *a, **k: None)
        pp.start()
        self.addCleanup(pp.stop)

    @staticmethod
    def _drain_queue() -> list:
        out = []
        try:
            while True:
                out.append(rag._event_q.get_nowait())
        except Exception:
            pass
        return out


# ── the shipped list ─────────────────────────────────────────────────────
class ShippedListTests(unittest.TestCase):
    def test_config_ships_the_list(self):
        shipped = _shipped_excludes()
        self.assertIsInstance(shipped, list)
        self.assertTrue(all(isinstance(p, str) and p.strip() for p in shipped))

    def test_original_indexer_defaults_are_kept(self):
        shipped = _shipped_excludes()
        for pat in _ORIGINAL_INDEXER_DEFAULTS:
            self.assertIn(pat, shipped)

    def test_indexer_default_comes_from_config(self):
        # One copy of the list, not two that drift: the indexer's module
        # default is core.config's value (whatever settings made it).
        self.assertEqual(list(rag.RAG_EXCLUDE_GLOBS),
                         list(config.RAG_EXCLUDE_GLOBS))

    def test_override_replaces_rule_is_documented(self):
        with open(os.path.join(_ROOT, "core", "config.py"),
                  encoding="utf-8") as f:
            src = f.read()
        block = src[:src.index("RAG_EXCLUDE_GLOBS = [")]
        block = block[block.rindex("# RAG_EXCLUDE_GLOBS"):]
        self.assertIn("REPLACES", block)


# ── the name tables ──────────────────────────────────────────────────────
class SecretNameTableTests(_ExcludeBase):
    def setUp(self):
        super().setUp()
        rag.RAG_EXCLUDE_GLOBS = _shipped_excludes()
        rag.RAG_INDEX_PATHS = [_WIN_ROOT, _POSIX_ROOT]

    def test_secret_shaped_names_are_excluded(self):
        for name in _EXCLUDED_NAMES:
            for root in (_WIN_ROOT, _POSIX_ROOT):
                for path in _spellings(root, name):
                    with self.subTest(path=path):
                        self.assertTrue(rag._is_excluded(path))

    def test_ordinary_names_are_included(self):
        for name in _INCLUDED_NAMES:
            for root in (_WIN_ROOT, _POSIX_ROOT):
                for path in _spellings(root, name):
                    with self.subTest(path=path):
                        self.assertFalse(rag._is_excluded(path))

    def test_secret_name_in_a_subfolder_still_excluded(self):
        for rel in ("Clients/Acme/passwords.txt",
                    r"Clients\Acme\level-devices.csv",
                    "Old/Archive/tls.key.txt"):
            for path in _spellings(_WIN_ROOT, rel):
                with self.subTest(path=path):
                    self.assertTrue(rag._is_excluded(path))

    def test_secret_named_folder_below_the_root_is_excluded(self):
        for rel in (r"Clients\Acme\Credentials\router.txt",
                    "Passwords/bank.md", "Recovery Keys/laptop.txt"):
            for root in (_WIN_ROOT, _POSIX_ROOT):
                for path in _spellings(root, rel):
                    with self.subTest(path=path):
                        self.assertTrue(rag._is_excluded(path))

    def test_folders_above_the_root_are_never_tested(self):
        # A user folder or a root that happens to contain "pass" / "token"
        # must not hide every file under it.
        root = r"C:\Users\Compass\Token Projects\Documents"
        rag.RAG_INDEX_PATHS = [root]
        for path in _spellings(root, "Clients/notes.md"):
            with self.subTest(path=path):
                self.assertFalse(rag._is_excluded(path))
        for path in _spellings(root, "Clients/passwords.txt"):
            with self.subTest(path=path):
                self.assertTrue(rag._is_excluded(path))

    def test_structural_excludes_any_case_either_slash(self):
        for path in (r"C:\Users\Someone\APPDATA\Local\cache\x.txt",
                     "c:/users/someone/appdata/roaming/app/x.md",
                     r"C:\Users\Someone\Documents\proj\NODE_MODULES\dep.js",
                     "/home/someone/docs/proj/.Git/config",
                     r"C:\Users\Someone\Documents\scratch.TMP"):
            with self.subTest(path=path):
                self.assertTrue(rag._is_excluded(path))


# ── the matcher itself ───────────────────────────────────────────────────
class MatcherRobustnessTests(_ExcludeBase):
    def test_owner_pattern_with_backslashes_matches(self):
        rag.RAG_INDEX_PATHS = []
        rag.RAG_EXCLUDE_GLOBS = [r"*\Private\*"]
        self.assertTrue(rag._is_excluded("C:/Users/Someone/private/notes.md"))
        self.assertTrue(rag._is_excluded(r"D:\Work\PRIVATE\plan.md"))
        self.assertFalse(rag._is_excluded(r"D:\Work\Public\plan.md"))

    def test_upper_case_pattern_matches_lower_case_name(self):
        rag.RAG_INDEX_PATHS = []
        rag.RAG_EXCLUDE_GLOBS = ["*.LOG", "*Journal*"]
        self.assertTrue(rag._is_excluded("/x/server.log"))
        self.assertTrue(rag._is_excluded(r"C:\x\my journal.md"))

    def test_not_under_any_root_tests_the_file_name_only(self):
        rag.RAG_INDEX_PATHS = [_WIN_ROOT]
        rag.RAG_EXCLUDE_GLOBS = ["*secret*"]
        self.assertFalse(rag._is_excluded(r"E:\Secret Stuff\notes.md"))
        self.assertTrue(rag._is_excluded(r"E:\Stuff\secret notes.md"))

    def test_junk_entries_are_ignored(self):
        rag.RAG_INDEX_PATHS = []
        rag.RAG_EXCLUDE_GLOBS = [None, 7, "", "   ", "*.tmp"]
        self.assertTrue(rag._is_excluded("/x/a.tmp"))
        self.assertFalse(rag._is_excluded("/x/a.md"))

    def test_a_bare_string_is_one_pattern_not_its_letters(self):
        # configure(rag_exclude_globs="*.tmp") must not become the patterns
        # "*", ".", "t", "m", "p" — the "*" would exclude everything.
        rag.RAG_INDEX_PATHS = []
        rag.RAG_EXCLUDE_GLOBS = "*.tmp"
        self.assertTrue(rag._is_excluded("/x/a.tmp"))
        self.assertFalse(rag._is_excluded("/x/a.md"))

    def test_runtime_change_takes_effect_at_once(self):
        rag.RAG_INDEX_PATHS = []
        rag.configure(rag_exclude_globs=["*.md"])
        self.assertTrue(rag._is_excluded("/x/a.md"))
        rag.configure(rag_exclude_globs=["*.txt"])
        self.assertFalse(rag._is_excluded("/x/a.md"))


# ── the walk, the watcher and the index agree ────────────────────────────
class _FakeVectors:
    def __init__(self, n):
        self._n = n

    def tolist(self):
        return [[0.1, 0.2, 0.3] for _ in range(self._n)]


class _FakeEmbedder:
    def encode(self, texts, **_):
        return _FakeVectors(len(list(texts)))


class _FakeCollection:
    """Just enough of a Chroma collection for _index_file / index_once."""

    def __init__(self):
        self.store = {}   # id -> metadata

    def get(self, where=None, include=None, limit=None, ids=None):
        items = list(self.store.items())
        if where and "file_id" in where:
            items = [(i, m) for i, m in items
                     if m.get("file_id") == where["file_id"]]
        if limit is not None:
            items = items[:limit]
        return {"ids": [i for i, _ in items],
                "metadatas": [m for _, m in items]}

    def delete(self, where=None, ids=None):
        if ids is not None:
            for i in ids:
                self.store.pop(i, None)
        elif where and "file_id" in where:
            for i in [i for i, m in self.store.items()
                      if m.get("file_id") == where["file_id"]]:
                self.store.pop(i, None)

    def add(self, ids=None, embeddings=None, documents=None, metadatas=None):
        for i, m in zip(ids, metadatas):
            self.store[i] = m

    def paths(self):
        return {os.path.basename(m["path"]) for m in self.store.values()}


class IndexAgreesWithExcludesTests(_ExcludeBase):
    def setUp(self):
        super().setUp()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = os.path.join(tmp.name, "docs")
        os.makedirs(self.root)
        rag.RAG_INDEX_PATHS = [self.root]
        # A custom list: the system tempdir sits under AppData\Local on
        # Windows, which the shipped list (rightly) skips.
        rag.RAG_EXCLUDE_GLOBS = ["*/node_modules/*", "*password*", "*.csv"]
        self.coll = _FakeCollection()
        rag._collection = self.coll
        rag._embed_model = _FakeEmbedder()
        p = mock.patch.object(rag, "is_available", return_value=True)
        p.start()
        self.addCleanup(p.stop)

    def _write(self, rel, text="some words"):
        full = os.path.join(self.root, *rel.split("/"))
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w", encoding="utf-8") as f:
            f.write(text)
        return full

    def _seed_chunk(self, path, chunks=1):
        fid = rag._file_id(path)
        for i in range(chunks):
            self.coll.store[f"{fid}:{i}"] = {
                "file_id": fid, "path": path, "sig": "old:1",
                "filename": os.path.basename(path), "chunk_index": i}

    def test_walk_skips_secret_files_and_folders(self):
        self._write("notes.md")
        self._write("Router PASSWORDS.txt")
        self._write("Passwords/bank.md")
        self._write("export.csv")
        got = sorted(os.path.basename(p) for p in rag._iter_files(self.root))
        self.assertEqual(got, ["notes.md"])

    def test_already_indexed_file_that_now_matches_is_dropped(self):
        # Indexed under an older list, still on disk, now excluded.
        keep = self._write("notes.md")
        secret = self._write("passwords.txt")
        in_folder = self._write("Passwords/bank.md")
        for p in (keep, secret, in_folder):
            self._seed_chunk(p)
        rag.index_once()
        self.assertEqual(self.coll.paths(), {"notes.md"})

    def test_a_new_runtime_exclude_drops_on_the_next_scan(self):
        journal = self._write("journal.md")
        rag.index_once()
        self.assertIn("journal.md", self.coll.paths())
        rag.configure(rag_exclude_globs=list(rag.RAG_EXCLUDE_GLOBS)
                      + ["*journal*"])
        rag.index_once()
        self.assertNotIn("journal.md", self.coll.paths())
        self.assertTrue(os.path.isfile(journal))   # dropped, never deleted

    def test_scan_summary_counts_the_dropped_files(self):
        # Counted in FILES, not chunks: one file with three chunks is 1.
        self._seed_chunk(self._write("passwords.txt"), chunks=3)
        self._seed_chunk(self._write("export.csv"))
        out = rag.index_once()
        self.assertEqual(out.get("excluded_dropped"), 2)
        self.assertEqual(self.coll.store, {})

    def test_non_excluded_existing_files_are_kept(self):
        # The pre-existing GC rule still holds: a file that exists and is not
        # excluded keeps its chunks even when this scan did not walk it.
        outside = os.path.join(os.path.dirname(self.root), "elsewhere.md")
        with open(outside, "w", encoding="utf-8") as f:
            f.write("x")
        self._seed_chunk(outside)
        rag.index_once()
        self.assertIn("elsewhere.md", self.coll.paths())


# ── search refuses excluded hits before the clean-up has run ─────────────
class _QueryVector(list):
    def tolist(self):
        return list(self)


class _QueryEmbedder:
    def encode(self, texts, **_):
        return [_QueryVector([0.1, 0.2, 0.3]) for _ in texts]


class _QueryCollection:
    """A Chroma collection whose query() returns every seeded chunk."""

    def __init__(self, paths):
        self.metas = [{"path": p, "filename": p.replace("\\", "/").rsplit(
            "/", 1)[-1], "chunk_index": 0, "ext": os.path.splitext(p)[1]}
            for p in paths]

    def query(self, query_embeddings=None, n_results=None, include=None):
        n = len(self.metas)
        return {"documents": [[f"chunk {i}" for i in range(n)]],
                "metadatas": [list(self.metas)],
                "distances": [[0.1 + 0.01 * i for i in range(n)]]}


class SearchNeverReturnsExcludedTests(_ExcludeBase):
    """index_once() drops an excluded file's chunks only in its clean-up pass
    AFTER the whole walk, which runs for an hour or more right after a big
    folder is added. Until then search itself must refuse those hits, or a
    passwords file indexed under an older list is still read out."""

    def setUp(self):
        super().setUp()
        rag.RAG_INDEX_PATHS = [_WIN_ROOT]
        rag.RAG_EXCLUDE_GLOBS = _shipped_excludes()
        self.keep = _WIN_ROOT + r"\meeting notes.md"
        self.journal = _WIN_ROOT + r"\Journal\monday.md"
        rag._collection = _QueryCollection([
            self.keep,
            _WIN_ROOT + r"\Router PASSWORDS.txt",
            _WIN_ROOT + r"\Clients\Acme\Credentials\wifi.md",
            _WIN_ROOT + r"\device export.csv",
            self.journal,
        ])
        rag._embed_model = _QueryEmbedder()
        for target, attr, value in ((rag, "is_available", lambda: True),
                                    (rag, "RAG_RERANKER_MODEL", ""),
                                    (rag, "_reranker", None)):
            p = mock.patch.object(target, attr, value)
            p.start()
            self.addCleanup(p.stop)

    def _hit_paths(self):
        return [h["path"] for h in rag.search("router", k=10)]

    def test_secret_hits_are_never_returned(self):
        self.assertEqual(self._hit_paths(), [self.keep, self.journal])

    def test_a_runtime_exclude_applies_before_any_scan(self):
        rag.configure(rag_exclude_globs=_shipped_excludes() + ["*journal*"])
        self.assertEqual(self._hit_paths(), [self.keep])


# ── the live watcher ─────────────────────────────────────────────────────
class WatcherRenameTests(_ExcludeBase):
    """A rename into an excluded name must not leave the old name's chunks
    searchable until the next full scan."""

    def setUp(self):
        super().setUp()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = tmp.name
        rag.RAG_INDEX_PATHS = [self.root]
        rag.RAG_EXCLUDE_GLOBS = ["*password*"]
        self.handler = self._start_fake_watchdog()
        self._drain_queue()

    def _start_fake_watchdog(self):
        import sys
        import types
        captured = {}

        class _Observer:
            def schedule(self, handler, root, recursive=True):
                captured["handler"] = handler

            def start(self):
                pass

        observers = types.ModuleType("watchdog.observers")
        observers.Observer = _Observer
        events = types.ModuleType("watchdog.events")
        events.FileSystemEventHandler = type("FileSystemEventHandler", (), {})
        with mock.patch.dict(sys.modules, {
                "watchdog": types.ModuleType("watchdog"),
                "watchdog.observers": observers,
                "watchdog.events": events}):
            self.assertTrue(rag._start_watchdog())
        return captured["handler"]

    def _event(self, src, dest=""):
        import types
        return types.SimpleNamespace(is_directory=False, src_path=src,
                                     dest_path=dest)

    def test_rename_to_a_secret_name_queues_only_the_old_name(self):
        old = os.path.join(self.root, "notes.txt")
        new = os.path.join(self.root, "Router PASSWORDS.txt")
        self.handler.on_any_event(self._event(old, new))
        # The old name no longer exists, so the drain deletes its chunks;
        # the new, excluded name is never queued for indexing.
        self.assertEqual(self._drain_queue(), [old])

    def test_ordinary_rename_queues_both_names(self):
        old = os.path.join(self.root, "draft.txt")
        new = os.path.join(self.root, "final.txt")
        self.handler.on_any_event(self._event(old, new))
        self.assertEqual(self._drain_queue(), [new, old])

    def test_edit_of_an_excluded_file_is_ignored(self):
        self.handler.on_any_event(self._event(
            os.path.join(self.root, "passwords.txt")))
        self.assertEqual(self._drain_queue(), [])

    def test_the_drain_deletes_the_queued_old_name(self):
        old = os.path.join(self.root, "notes.txt")   # never created: gone
        self.handler.on_any_event(self._event(
            old, os.path.join(self.root, "passwords.txt")))
        deleted = []
        flags = iter([False, True])         # exactly one pass of the loop
        clock = iter([0.0])                 # queued at 0, due at 2, now 100
        import types
        fake_time = types.SimpleNamespace(time=lambda: next(clock, 100.0))
        with mock.patch.object(rag, "_delete_file",
                               side_effect=lambda fid: deleted.append(fid)), \
             mock.patch.object(rag, "_index_file") as index_file, \
             mock.patch.object(rag._stop_flag, "is_set",
                               lambda: next(flags, True)), \
             mock.patch.object(rag, "time", fake_time):
            rag._drain_event_queue()
        self.assertEqual(deleted, [rag._file_id(old)])
        index_file.assert_not_called()


# ── an owner override replaces the list ──────────────────────────────────
class OwnerOverrideTests(unittest.TestCase):
    def _apply(self, payload):
        with mock.patch.object(config, "RAG_EXCLUDE_GLOBS",
                               _shipped_excludes()), \
             mock.patch.object(config, "_USER_SETTINGS_ERROR", None), \
             mock.patch.object(config, "_SAFETY_SETTINGS_WARNINGS", []), \
             mock.patch("os.path.exists", return_value=True), \
             mock.patch("builtins.open", mock.mock_open(read_data="{}")), \
             mock.patch("json.load", return_value=payload):
            config._apply_user_settings()
            return list(config.RAG_EXCLUDE_GLOBS)

    def test_saved_list_replaces_the_shipped_one(self):
        got = self._apply({"RAG_EXCLUDE_GLOBS": ["*.journal", "*/Private/*"]})
        self.assertEqual(got, ["*.journal", "*/Private/*"])

    def test_it_is_not_a_merged_safety_floor(self):
        self.assertNotIn("RAG_EXCLUDE_GLOBS", config._SAFETY_LIST_BASELINE)

    def test_a_non_list_value_keeps_the_shipped_list(self):
        got = self._apply({"RAG_EXCLUDE_GLOBS": "*.journal"})
        self.assertEqual(got, _shipped_excludes())


if __name__ == "__main__":
    unittest.main()
