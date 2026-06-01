#!/usr/bin/env python3
"""Build a clean, shareable JARVIS release from the git-tracked tree.

The release contains ONLY git-tracked files, so every gitignored thing —
personal memory, voiceprints, credentials, .env, logs, backups, the personal
skills — is excluded BY CONSTRUCTION (not by a fragile copy-filter). The
PII/secret leak scan then runs on the OUTPUT as a hard gate; if anything
sensitive slipped in, the build aborts and removes the artifact.

    python tools/build_release.py           # -> dist/jarvis-<VERSION>.zip
    python tools/build_release.py --keep     # also leave the unzipped dir

Exit 0 = built + leak-scan clean; 1 = leak found / git error.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import zipfile

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _version() -> str:
    try:
        with open(os.path.join(_ROOT, "VERSION"), encoding="utf-8") as f:
            return f.read().strip() or "0.0.0-dev"
    except OSError:
        return "0.0.0-dev"


def main(argv: list[str]) -> int:
    keep = "--keep" in argv
    version = _version()
    dist = os.path.join(_ROOT, "dist")
    name = f"jarvis-{version}"
    out_dir = os.path.join(dist, name)

    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)

    # Export ONLY git-tracked files (so .gitignore is the single source of
    # truth for what ships — no personal data can leak through a copy-glob).
    try:
        res = subprocess.run(["git", "ls-files"], cwd=_ROOT,
                             capture_output=True, text=True, timeout=60, check=True)
    except Exception as e:  # noqa: BLE001
        print(f"[build] git ls-files failed: {e}")
        return 1
    tracked = [l.strip() for l in res.stdout.splitlines() if l.strip()]
    for rel in tracked:
        src = os.path.join(_ROOT, rel)
        dst = os.path.join(out_dir, rel)
        if not os.path.isfile(src):
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        try:
            shutil.copy2(src, dst)
        except OSError as e:  # noqa: BLE001
            print(f"[build] warn: could not copy {rel}: {e}")
    print(f"[build] exported {len(tracked)} tracked files -> {out_dir}")

    # Hard leak gate on the OUTPUT (HARD findings only; benign WARNs — generic
    # example IPs, fake test fixtures — are allowed).
    chk = subprocess.run(
        [sys.executable, os.path.join(_ROOT, "tools", "check_no_pii.py"), out_dir],
        capture_output=True, text=True, timeout=300)
    tail = "\n".join(chk.stdout.splitlines()[-6:])
    print(tail)
    if chk.returncode != 0:
        print("[build] LEAK SCAN FAILED on the release output — aborting (not shipping).")
        shutil.rmtree(out_dir, ignore_errors=True)
        return 1

    # Zip it (top-level folder = the release name).
    zip_path = os.path.join(dist, f"{name}.zip")
    if os.path.exists(zip_path):
        os.remove(zip_path)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _dirs, files in os.walk(out_dir):
            for fn in files:
                ap = os.path.join(root, fn)
                arc = os.path.join(name, os.path.relpath(ap, out_dir))
                z.write(ap, arc)
    size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"[build] wrote {zip_path}  ({size_mb:.1f} MB, {len(tracked)} files)")

    if not keep:
        shutil.rmtree(out_dir, ignore_errors=True)
    print(f"[build] OK — jarvis {version} release built + leak-scan clean.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
