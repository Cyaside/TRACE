"""Validate Kaggle's extracted dataset layout before any install or training."""
from __future__ import annotations

import json
from pathlib import Path
import zipfile

from src.common import digest


def _verify_prepared(root, manifest_hash, data_hash):
    root = Path(root)
    if digest(root / "manifest.json") != manifest_hash:
        raise ValueError("Prepared manifest checksum mismatch")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest["data_hash"] != data_hash:
        raise ValueError("Prepared dataset identity mismatch")
    for name, expected in manifest["files"].items():
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Prepared filename outside dataset root")
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"Prepared file missing or checksum mismatch: {name}")
    return root


def resolve_prepared(input_root, *, manifest_hash, data_hash, archive_hash, extract_root):
    """Accept auto-extracted input or an intact ZIP; verify identical file bytes.

    Kaggle normally expands prepared.zip to prepared/ on upload. An intact ZIP
    is supported for other environments, with archive and member checksums.
    """
    input_root = Path(input_root)
    candidates = [p.parent for p in input_root.rglob("manifest.json")
                  if digest(p) == manifest_hash]
    if len(candidates) > 1:
        raise ValueError("More than one matching prepared dataset is attached")
    if candidates:
        return _verify_prepared(candidates[0], manifest_hash, data_hash)
    archives = [p for p in input_root.rglob("prepared.zip") if digest(p) == archive_hash]
    if len(archives) > 1:
        raise ValueError("More than one matching prepared archive is attached")
    if archives:
        root = Path(extract_root)
        with zipfile.ZipFile(archives[0]) as archive:
            manifest_bytes = archive.read("manifest.json")
            import hashlib
            if hashlib.sha256(manifest_bytes).hexdigest() != manifest_hash:
                raise ValueError("Archived manifest checksum mismatch")
            manifest = json.loads(manifest_bytes)
            allowed = {"manifest.json", *manifest["files"]}
            if set(archive.namelist()) != allowed:
                raise ValueError("Unexpected prepared archive members")
            for name in allowed:
                if not (root / name).resolve().is_relative_to(root.resolve()):
                    raise ValueError("Archive filename outside dataset root")
            archive.extractall(root)
        return _verify_prepared(root, manifest_hash, data_hash)
    inventory = sorted(str(p.relative_to(input_root)) for p in input_root.rglob("*") if p.is_file())[:40]
    raise FileNotFoundError("Matching prepared manifest/ZIP not found. Attached files: " + repr(inventory))
