#!/usr/bin/env python3
"""Download KITE's public Google Drive assets and restore their repository paths."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tarfile
import tempfile
import time

ROOT = Path(__file__).resolve().parent


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_path(root, relative):
    parts = PurePosixPath(relative).parts
    if not parts or relative != PurePosixPath(relative).as_posix() or "\\" in relative:
        raise ValueError(f"Invalid asset path: {relative!r}")
    if PurePosixPath(relative).is_absolute() or any(p in {"..", ".git", ".cache"} for p in parts):
        raise ValueError(f"Unsafe asset path: {relative!r}")
    target = root.joinpath(*parts)
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Asset path escapes destination: {relative}")
    for parent in [target, *target.parents]:
        if parent == root:
            break
        if parent.is_symlink():
            raise ValueError(f"Refusing asset path through symlink: {target}")
    return target


def matches(path, record):
    return path.is_file() and path.stat().st_size == record["size"] and sha256(path) == record["sha256"]


def load_manifest(path):
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported asset manifest schema")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", manifest["release"]):
        raise ValueError("Invalid release name")
    seen = set()
    names = set()
    for bundle in manifest["bundles"]:
        if bundle["name"] in names:
            raise ValueError("Duplicate bundle name")
        names.add(bundle["name"])
        if Path(bundle["archive"]).name != bundle["archive"] or not bundle["archive"].endswith(".tar.gz"):
            raise ValueError("Archive name must be a plain filename")
        for record in [bundle, *bundle["files"]]:
            if not isinstance(record["size"], int) or record["size"] < 0:
                raise ValueError("Invalid size in manifest")
            if not re.fullmatch(r"[0-9a-f]{64}", record["sha256"]):
                raise ValueError("Invalid SHA-256 in manifest")
        archive_paths = set()
        for record in bundle["files"]:
            safe_path(Path("/tmp/kite-manifest-validation"), record["path"])
            archive_path = record.get("archive_path", record["path"])
            safe_path(Path("/tmp/kite-manifest-validation"), archive_path)
            if archive_path in archive_paths:
                raise ValueError(f"Duplicate archive member: {archive_path}")
            archive_paths.add(archive_path)
            if record["path"] in seen:
                raise ValueError(f"Duplicate asset: {record['path']}")
            seen.add(record["path"])
    return manifest


def download(bundle, cache):
    archive = cache / bundle["archive"]
    if archive.exists():
        if matches(archive, bundle):
            return archive
        raise ValueError(f"Cached archive has a different checksum; move it aside: {archive}")
    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError("Install the downloader first: python -m pip install -r requirements-download.txt") from exc
    if not re.fullmatch(r"[A-Za-z0-9_-]+", bundle["drive_file_id"]):
        raise ValueError(f"Missing Google Drive file ID for {bundle['name']}")
    partial = archive.with_name(archive.name + ".partial")
    for attempt in range(3):
        try:
            # Public downloads only: do not read the user's Google cookies.
            result = gdown.download(id=bundle["drive_file_id"], output=str(partial),
                                    quiet=False, use_cookies=False, resume=True)
            if not result or not matches(partial, bundle):
                raise ValueError(f"Download failed checksum/size verification: {bundle['name']}")
            partial.replace(archive)
            return archive
        except Exception as exc:
            if attempt == 2:
                raise RuntimeError(f"Could not download {bundle['name']}: {exc}. "
                                   "Check the public Drive link or retry later. "
                                   f"If verification repeatedly fails, move {partial} aside before retrying.") from exc
            print("Download interrupted; retrying. Drive may temporarily limit popular files.", file=sys.stderr)
            time.sleep(2 ** attempt)
    raise RuntimeError("Unreachable")


def resolve_drive_ids(manifest, bundles):
    """Allow a maintainer to upload the named archives without editing the script."""
    unresolved = [b for b in bundles if not b.get("drive_file_id")]
    if not unresolved:
        return
    folder = manifest.get("drive_folder_url", "")
    if not re.fullmatch(r"https://drive\.google\.com/drive/folders/[A-Za-z0-9_-]+", folder):
        raise ValueError("Manifest needs a public Google Drive folder URL or file IDs")
    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError("Install the downloader first: python -m pip install -r requirements-download.txt") from exc
    try:
        listing = gdown.download_folder(url=folder, skip_download=True, quiet=True, use_cookies=False)
    except Exception as exc:
        raise RuntimeError(f"Could not list the public release folder: {folder}: {exc}") from exc
    for bundle in unresolved:
        candidates = [entry for entry in (listing or []) if Path(entry.path).name == bundle["archive"]]
        if len(candidates) != 1:
            raise FileNotFoundError(f"Expected exactly one {bundle['archive']} in {folder}; "
                                    f"found {len(candidates)}. The release upload may not be complete.")
        bundle["drive_file_id"] = candidates[0].id


def install_archive(archive, bundle, root, cache):
    if not matches(archive, bundle):
        raise ValueError(f"Archive checksum/size mismatch: {archive}")
    expected = {record.get("archive_path", record["path"]): record for record in bundle["files"]}
    if len(expected) != len(bundle["files"]):
        raise ValueError("Duplicate archive members in manifest")
    with tempfile.TemporaryDirectory(prefix="extract-", dir=cache) as temporary:
        stage = Path(temporary)
        seen = set()
        with tarfile.open(archive, "r:gz") as tar:
            for member in tar:
                if not member.isfile() or member.name not in expected or member.name in seen:
                    raise ValueError(f"Unexpected, duplicate, or nonregular archive member: {member.name}")
                record = expected[member.name]
                if member.size != record["size"]:
                    raise ValueError(f"Wrong member size: {member.name}")
                safe_path(stage, member.name)
                target = safe_path(stage, record["path"])
                target.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output)
                if not matches(target, record):
                    raise ValueError(f"Member checksum mismatch: {member.name}")
                seen.add(member.name)
        if seen != set(expected):
            raise ValueError(f"Archive is missing {len(set(expected) - seen)} assets")
        # All archive members are verified before any asset is installed.
        for record in expected.values():
            target = safe_path(root, record["path"])
            if target.exists() and not matches(target, record):
                raise FileExistsError(f"Preserving different local file: {target}")
        for record in expected.values():
            relative = record["path"]
            target = safe_path(root, relative)
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            # Atomic, no-clobber install; staging is on the destination filesystem.
            try:
                os.link(stage / relative, target)
            except FileExistsError:
                if not matches(target, record):
                    raise
    return len(expected)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "asset_downloads.json")
    parser.add_argument("--root", type=Path, default=ROOT, help="Destination repository root")
    parser.add_argument("--only", nargs="+", help="Bundle names (default: all)")
    parser.add_argument("--verify-only", action="store_true", help="Check local files without downloading")
    parser.add_argument("--archive-dir", type=Path, help="Use manually downloaded archives instead of the network")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    manifest = load_manifest(args.manifest)
    bundles = manifest["bundles"]
    if args.only:
        unknown = set(args.only) - {b["name"] for b in bundles}
        if unknown:
            parser.error(f"Unknown bundles: {', '.join(sorted(unknown))}")
        bundles = [b for b in bundles if b["name"] in args.only]
    missing = []
    conflicts = []
    pending = []
    for bundle in bundles:
        needed = False
        for record in bundle["files"]:
            path = safe_path(root, record["path"])
            if not path.exists():
                missing.append(record["path"])
                needed = True
            elif not matches(path, record):
                conflicts.append(record["path"])
        if needed:
            pending.append(bundle)
        else:
            print(f"{bundle['name']}: {len(bundle['files'])} local files checked")
    if conflicts:
        raise FileExistsError("Preserving different local files; move them aside before retrying:\n" + "\n".join(conflicts))
    if args.verify_only:
        if missing:
            raise FileNotFoundError(f"{len(missing)} missing assets:\n" + "\n".join(missing[:20]))
        print("All selected assets match their SHA-256 checksums.")
        return 0
    if pending:
        if not args.archive_dir:
            resolve_drive_ids(manifest, pending)
        root.mkdir(parents=True, exist_ok=True)
        cache = root / ".cache" / "kite-assets" / manifest["release"]
        cache.mkdir(parents=True, exist_ok=True)
        for bundle in pending:
            archive = args.archive_dir / bundle["archive"] if args.archive_dir else download(bundle, cache)
            count = install_archive(archive, bundle, root, cache)
            print(f"{bundle['name']}: installed/verified {count} files", flush=True)
    print("KITE assets are ready. Training demonstrations are generated locally; see README.md.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError, tarfile.TarError) as exc:
        print(f"Asset setup failed: {exc}", file=sys.stderr)
        sys.exit(1)
