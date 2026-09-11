#!/usr/bin/env python3
"""Download and unpack the ExtraSensory files this project needs.

Dataset: ExtraSensory (Vaizman, Ellis & Lanckriet, IEEE Pervasive Computing
16(4), 2017), http://extrasensory.ucsd.edu/. Cited at point of use; the raw
data itself is NOT committed to this repository, per the challenge rules.

The site asks that the large raw archives be downloaded one at a time, so this
script is deliberately sequential and resumable (HTTP Range). Total for the
required set is ~15 GB; budget the disk and the wall-clock time.

    python scripts/fetch_extrasensory.py --only original_labels features_labels
    python scripts/fetch_extrasensory.py --all
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path
from urllib import request

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ats.config import load_config, resolve  # noqa: E402

CHUNK = 1 << 20


def download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    existing = dest.stat().st_size if dest.exists() else 0
    req = request.Request(url, headers={"User-Agent": "ask-the-sensors/0.1"})
    if existing:
        req.add_header("Range", f"bytes={existing}-")
        print(f"  resuming at {existing / 1e6:.1f} MB")
    try:
        with request.urlopen(req) as response, open(dest, "ab" if existing else "wb") as out:
            total = int(response.headers.get("Content-Length", 0)) + existing
            done = existing
            while True:
                chunk = response.read(CHUNK)
                if not chunk:
                    break
                out.write(chunk)
                done += len(chunk)
                if total:
                    pct = 100.0 * done / total
                    print(f"\r  {done / 1e6:8.1f} / {total / 1e6:.1f} MB ({pct:5.1f}%)",
                          end="", flush=True)
        print()
    except Exception as exc:  # noqa: BLE001
        print(f"\n  FAILED: {exc}\n  Re-run to resume from {dest.stat().st_size if dest.exists() else 0} bytes.")
        raise
    return dest


def unpack(archive: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    print(f"  unpacking into {target}")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(target)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="*", default=None,
                        help="subset of file keys from configs/data.yaml")
    parser.add_argument("--all", action="store_true", help="fetch every required file")
    parser.add_argument("--keep-archives", action="store_true",
                        help="do not delete the .zip after unpacking")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_config("data")
    files = cfg["dataset"]["files"]
    root = resolve(cfg["dataset"]["root"])

    if args.only:
        keys = [k for k in args.only if k in files]
        unknown = set(args.only) - set(files)
        if unknown:
            print(f"unknown keys ignored: {sorted(unknown)}")
    elif args.all:
        keys = [k for k, v in files.items() if v.get("required", True)]
    else:
        parser.error("pass --all or --only KEY [KEY ...]")

    total_gb = sum(float(files[k].get("approx_gb", 0)) for k in keys)
    print(f"Fetching {len(keys)} file(s), about {total_gb:.1f} GB, into {root}")
    if args.dry_run:
        for k in keys:
            print(f"  {k:18s} {files[k]['url']}")
        return 0

    for key in keys:
        spec = files[key]
        print(f"\n[{key}] {spec['url']}")
        archive = root / "_archives" / Path(spec["url"]).name
        download(spec["url"], archive)
        unpack(archive, root / key)
        if not args.keep_archives:
            archive.unlink(missing_ok=True)
            print("  archive removed (re-run this script to fetch it again)")

    print("\nDone. Raw data lives under data/raw/ and is git-ignored by design.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
