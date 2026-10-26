from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "初始研究"
AUDIT = ROOT / "flash 审计修复"
MANIFEST = AUDIT / "audit" / "migration_manifest.json"
EXCLUDED = {ARCHIVE.name, AUDIT.name}
SENSITIVE_NAMES = {".env", "credentials.env", "id_rsa", "id_ed25519"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sensitive(path: Path) -> bool:
    return any(part in SENSITIVE_NAMES for part in path.parts) or path.suffix in {".pem", ".key"}


def inventory(root: Path, archive_mode: bool = False) -> dict:
    files = {}
    top_entries = []
    for entry in sorted(root.iterdir(), key=lambda item: item.name):
        if not archive_mode and entry.name in EXCLUDED:
            continue
        top_entries.append(entry.name)
        paths = [entry] if entry.is_symlink() or entry.is_file() else sorted(entry.rglob("*"))
        for path in paths:
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                files[relative] = {"kind": "symlink", "target": os.readlink(path)}
            elif path.is_file():
                record = {"kind": "file", "bytes": path.stat().st_size}
                if sensitive(path):
                    record["sha256"] = None
                    record["hash_skipped"] = "sensitive_filename"
                else:
                    record["sha256"] = sha256(path)
                files[relative] = record
    return {"root": str(root), "top_entries": top_entries, "files": files}


def compare(before: dict, after: dict) -> None:
    if before["top_entries"] != after["top_entries"]:
        raise AssertionError("Top-level entries differ after migration")
    if before["files"] != after["files"]:
        before_keys = set(before["files"])
        after_keys = set(after["files"])
        raise AssertionError(
            "Migrated content differs: "
            f"missing={sorted(before_keys - after_keys)[:5]}, "
            f"extra={sorted(after_keys - before_keys)[:5]}"
        )


def migrate() -> None:
    if not ARCHIVE.is_dir() or any(ARCHIVE.iterdir()):
        raise RuntimeError("初始研究 must be an existing empty directory")
    before = inventory(ROOT)
    before["created_at"] = datetime.now(timezone.utc).isoformat()
    before["old_root"] = str(ROOT)
    before["archive_root"] = str(ARCHIVE)
    AUDIT.mkdir(exist_ok=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(
        json.dumps({"status": "hashing_complete_before_move", **before}, ensure_ascii=False, indent=2) + "\n"
    )
    moved = []
    try:
        for name in before["top_entries"]:
            source = ROOT / name
            destination = ARCHIVE / name
            if destination.exists() or destination.is_symlink():
                raise FileExistsError(destination)
            os.rename(source, destination)
            moved.append((source, destination))
    except Exception:
        for source, destination in reversed(moved):
            if destination.exists() or destination.is_symlink():
                if source.exists() or source.is_symlink():
                    raise RuntimeError(f"Cannot roll back {destination} because {source} exists")
                os.rename(destination, source)
        raise
    after = inventory(ARCHIVE, archive_mode=True)
    compare(before, after)
    remaining = sorted(path.name for path in ROOT.iterdir())
    if remaining != sorted(EXCLUDED):
        raise AssertionError(f"Unexpected entries left in original root: {remaining}")
    manifest = {
        "status": "passed",
        "created_at": before["created_at"],
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "old_root": str(ROOT),
        "archive_root": str(ARCHIVE),
        "top_entries": before["top_entries"],
        "file_count": len(before["files"]),
        "total_bytes": sum(item.get("bytes", 0) for item in before["files"].values()),
        "files": before["files"],
        "historical_path_note": (
            "Absolute paths inside frozen manifests and executed SQL are historical provenance. "
            "Their file bytes are unchanged; this manifest maps the old root to 初始研究."
        ),
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: manifest[key] for key in ["status", "top_entries", "file_count", "total_bytes"]}, ensure_ascii=False))


def verify() -> None:
    manifest = json.loads(MANIFEST.read_text())
    if manifest["status"] != "passed":
        raise RuntimeError("Migration did not complete")
    before = {"top_entries": manifest["top_entries"], "files": manifest["files"]}
    after = inventory(ARCHIVE, archive_mode=True)
    compare(before, after)
    remaining = sorted(path.name for path in ROOT.iterdir())
    if remaining != sorted(EXCLUDED):
        raise AssertionError(f"Unexpected entries left in original root: {remaining}")
    print(json.dumps({"status": "passed", "file_count": len(after["files"]), "remaining": remaining}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    verify() if args.verify else migrate()
