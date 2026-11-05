"""Verify the reviewed archive against the current checkout's frozen sources.

The demand/early-service study (and its parent fine-tuning protocol) pins a
SHA-256 for every frozen source file and for every model checkpoint. A checkout
produced on a different platform can differ from the archive ONLY by line
endings (the archive was packaged from a CRLF/Windows checkout). This helper:

  * checks the archive's own integrity hash,
  * verifies all 12 model checkpoints against the manifest,
  * compares every frozen source file to the pinned hash,
  * repairs ONLY line-ending differences in the working tree (with
    --repair-line-endings), rewriting a file to the exact frozen bytes,
  * REFUSES any file whose content differs beyond line endings, printing a diff
    summary instead of touching it.

It never edits a file whose normalized content does not match the frozen source,
so it cannot mask a real code change. Run from the repository root.

    python scripts/check_frozen_checkpoints.py [--repair-line-endings]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "runs" / "pedestrian-finetune-cluster-results.zip"
ARCHIVE_SHA = "255b82536b3a341c8964ef2338b5bd0a5e54fbe73dfd8e89fd9315b18c46354b"
PREFIX = "runs/pedestrian-finetune-cluster/"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalized_matches(current: bytes, frozen_sha: str) -> bytes | None:
    """Return the byte content that hashes to frozen_sha if the only difference
    is line endings; otherwise None. We try to reproduce the frozen bytes from
    the current content by normalizing to LF then expanding to CRLF, so a repair
    writes the EXACT frozen bytes (not a guess)."""
    lf = current.replace(b"\r\n", b"\n")
    for candidate in (current, lf, lf.replace(b"\n", b"\r\n")):
        if sha256(candidate) == frozen_sha:
            return candidate
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repair-line-endings", action="store_true",
                        help="rewrite working-tree files that differ ONLY by line endings")
    args = parser.parse_args()

    if not ARCHIVE.exists():
        print(f"FAIL: archive not found at {ARCHIVE.relative_to(ROOT)}")
        return 1
    actual = sha256(ARCHIVE.read_bytes())
    if actual != ARCHIVE_SHA:
        print(f"FAIL: archive hash mismatch\n  expected {ARCHIVE_SHA}\n  actual   {actual}")
        return 1
    print(f"archive hash OK ({ARCHIVE.relative_to(ROOT)})")

    with zipfile.ZipFile(ARCHIVE) as z:
        manifest = json.loads(z.read("manifest.json"))
        parent = json.loads(z.read(PREFIX + "protocol.json"))
        # Model checkpoints: hash every mappo-*-r* final.pt against the manifest.
        checkpoints = sorted(m for m in manifest
                             if m.startswith(PREFIX + "models/") and m.endswith("final.pt"))
        bad_ckpt = [m for m in checkpoints if sha256(z.read(m)) != manifest[m]]
        if bad_ckpt:
            print("FAIL: checkpoint hash mismatch:\n  " + "\n  ".join(bad_ckpt))
            return 1
        print(f"checkpoints OK ({len(checkpoints)} verified)")

    frozen = parent["source"]
    missing, repaired, repairable, content_diff = [], [], [], []
    for name, frozen_sha in sorted(frozen.items()):
        path = ROOT / name
        if not path.exists():
            missing.append(name)
            continue
        current = path.read_bytes()
        if sha256(current) == frozen_sha:
            continue
        exact = normalized_matches(current, frozen_sha)
        if exact is None:
            content_diff.append(name)
        elif args.repair_line_endings:
            path.write_bytes(exact)
            repaired.append(name)
        else:
            repairable.append(name)

    if missing:
        print("FAIL: frozen source files absent from checkout:\n  " + "\n  ".join(missing))
    if content_diff:
        print("FAIL: frozen source files differ beyond line endings (REFUSING to touch):")
        for name in content_diff:
            print(f"  {name}  (real code change vs the frozen snapshot)")
    if repairable:
        print("Line-ending differences found (rerun with --repair-line-endings to fix):")
        for name in repairable:
            print(f"  {name}")
    if repaired:
        print(f"repaired line endings in {len(repaired)} file(s):")
        for name in repaired:
            print(f"  {name}")

    ok = not missing and not content_diff and not repairable
    total = len(frozen)
    if ok:
        print(f"\nPASS: {len(checkpoints)} checkpoints and {total} frozen source files verified.")
        return 0
    print(f"\nNOT PASSING: {total} frozen source files checked; see issues above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
