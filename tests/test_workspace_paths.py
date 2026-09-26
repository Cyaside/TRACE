"""Private output isolation and relocation contracts; no model training."""
from __future__ import annotations

import dataclasses
import datetime as dt
import json
import subprocess

import pytest

from src import common
from src.common import digest, object_hash, reports_dir
from src.training import ledger
from src.training.neural import TrainConfig


def test_explicit_report_date_stays_inside_private_workspace(tmp_path):
    expected = tmp_path / ".local" / "reports" / "2026-10-03"
    assert reports_dir(tmp_path, "2026-10-03") == expected
    assert reports_dir(tmp_path, dt.date(2026, 10, 3)) == expected
    assert not expected.exists(), "Resolving a path must not create evidence"


@pytest.mark.parametrize("date", [
    "../2026-10-03", "2026-10-03/../../public", "2026-02-30", "not-a-date",
])
def test_invalid_report_date_cannot_escape_private_workspace(tmp_path, date):
    with pytest.raises(ValueError):
        reports_dir(tmp_path, date)
    assert list(tmp_path.iterdir()) == []


def test_automatic_report_date_uses_jakarta_offset(tmp_path, monkeypatch):
    class FixedDateTime(dt.datetime):
        @classmethod
        def now(cls, tz=None):
            instant = cls(2026, 10, 3, 17, 30, tzinfo=dt.timezone.utc)
            return instant.astimezone(tz) if tz is not None else instant.replace(tzinfo=None)

    monkeypatch.setattr(common.dt, "datetime", FixedDateTime)
    assert reports_dir(tmp_path) == tmp_path / ".local" / "reports" / "2026-10-04"


def test_ledger_appends_only_to_private_dated_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(ledger, "ROOT", tmp_path)
    monkeypatch.setattr(ledger, "reports_dir", lambda root: reports_dir(root, "2026-10-03"))
    ledger.append_run({"run_id": "fixture", "status": "started"})
    ledger.append_run({"run_id": "fixture", "status": "complete"})

    expected = reports_dir(tmp_path, "2026-10-03") / "experiment-ledger.jsonl"
    assert [path for path in tmp_path.rglob("*") if path.is_file()] == [expected]
    records = [json.loads(line) for line in expected.read_text(encoding="utf-8").splitlines()]
    assert [record["status"] for record in records] == ["started", "complete"]
    assert all(record["run_id"] == "fixture" for record in records)
    assert all(dt.datetime.fromisoformat(record["recorded_at_utc"]).utcoffset() == dt.timedelta(0)
               for record in records)


def test_provenance_reads_private_environment_and_config_without_commit(tmp_path, monkeypatch):
    monkeypatch.setattr(ledger, "ROOT", tmp_path)

    def unborn_revision(command, **kwargs):
        assert command == ["git", "rev-parse", "HEAD"]
        assert kwargs["cwd"] == tmp_path
        raise subprocess.CalledProcessError(128, command)

    monkeypatch.setattr(ledger.subprocess, "check_output", unborn_revision)
    lock = tmp_path / ".local" / "environment" / "environment-lock.txt"
    lock.parent.mkdir(parents=True)
    lock.write_text("private-fixture-environment\n", encoding="utf-8")
    obsolete_lock = tmp_path / "reports" / "environment-lock.txt"
    obsolete_lock.parent.mkdir()
    obsolete_lock.write_text("obsolete-public-path\n", encoding="utf-8")
    private_config = tmp_path / ".local" / "configs" / "study.yaml"
    private_config.parent.mkdir()
    private_config.write_text("cohort: fixture\n", encoding="utf-8")
    reusable_config = tmp_path / "configs" / "reusable.yaml"
    reusable_config.parent.mkdir()
    reusable_config.write_text("seed: 42\n", encoding="utf-8")

    evidence = ledger.provenance({"cohort_hash": "fixture-cohort"})
    expected_code = {path.relative_to(tmp_path).as_posix(): digest(path)
                     for path in (private_config, reusable_config)}
    assert evidence["code_commit"] is None
    assert evidence["environment_lock"] == digest(lock)
    assert evidence["environment_lock"] != digest(obsolete_lock)
    assert evidence["code_content_hash"] == object_hash(expected_code)
    assert evidence["cohort_hash"] == "fixture-cohort"
    private_config.write_text("cohort: changed-fixture\n", encoding="utf-8")
    assert ledger.provenance()["code_content_hash"] != evidence["code_content_hash"]


def test_neural_output_relocation_preserves_experiment_identity(tmp_path):
    config = TrainConfig()
    assert config.output_root == str(common.ROOT / "artifacts" / "runs")
    relocated = dataclasses.replace(config, output_root=str(tmp_path / "relocated-runs"))
    assert config.experiment_dict() == relocated.experiment_dict()
    assert object_hash(config.experiment_dict()) == object_hash(relocated.experiment_dict())
