"""Cloud input regressions: Kaggle extracts archives before attaching them."""
import json
import zipfile

import pytest

from src.common import digest, object_hash, write_json
from src.training.kaggle_runtime import resolve_prepared


def fixture(root):
    root.mkdir(parents=True)
    # Binary contents stand in for Parquet; this test checks transport integrity.
    (root / "requests.parquet").write_bytes(b"PAR1\x00\xff")
    (root / "vocabulary.json").write_text('{"items": [20]}', encoding="utf-8")
    files = {name: digest(root / name) for name in ("requests.parquet", "vocabulary.json")}
    manifest = {"files": files, "data_hash": object_hash(files)}
    write_json(root / "manifest.json", manifest)
    return {"manifest_hash": digest(root / "manifest.json"), "data_hash": manifest["data_hash"]}


def test_kaggle_auto_extracted_layout_and_immutable_input(tmp_path):
    attached = tmp_path / "input" / "trace-v2-inputs-20261010" / "prepared"
    kwargs = fixture(attached)
    found = resolve_prepared(tmp_path / "input", **kwargs, archive_hash="not-present",
                             extract_root=tmp_path / "working")
    assert found == attached
    assert not (tmp_path / "working").exists()  # Read-only mount needs no copy.
    (attached / "requests.parquet").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="requests.parquet"):
        resolve_prepared(tmp_path / "input", **kwargs, archive_hash="not-present",
                         extract_root=tmp_path / "working")


def test_intact_zip_fallback_preserves_file_hashes(tmp_path):
    source = tmp_path / "source"
    kwargs = fixture(source)
    attached = tmp_path / "input" / "dataset"
    attached.mkdir(parents=True)
    archive = attached / "prepared.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        for path in source.iterdir():
            handle.write(path, path.name)
    target = tmp_path / "working" / "trace-data"
    found = resolve_prepared(tmp_path / "input", **kwargs, archive_hash=digest(archive), extract_root=target)
    assert found == target
    for path in source.iterdir():
        assert digest(target / path.name) == digest(path)


def test_missing_ambiguous_and_invalid_archive_report_before_training(tmp_path):
    first = tmp_path / "input" / "dataset" / "prepared"
    kwargs = fixture(first)
    with pytest.raises(FileNotFoundError, match="Attached files"):
        resolve_prepared(tmp_path / "input", **{**kwargs, "manifest_hash": "wrong"},
                         archive_hash="wrong", extract_root=tmp_path / "out")
    fixture(tmp_path / "input" / "duplicate" / "prepared")
    with pytest.raises(ValueError, match="More than one"):
        resolve_prepared(tmp_path / "input", **kwargs, archive_hash="wrong", extract_root=tmp_path / "out")
    bad_input = tmp_path / "bad-input"
    bad_input.mkdir()
    archive = bad_input / "prepared.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        for path in first.iterdir():
            handle.write(path, path.name)
        handle.writestr("../unexpected.txt", "forbidden")
    with pytest.raises(ValueError, match="Unexpected prepared archive members"):
        resolve_prepared(bad_input, **kwargs, archive_hash=digest(archive), extract_root=tmp_path / "out")
