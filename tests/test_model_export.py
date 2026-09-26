"""Weights exports retain tensors and exclude private training evidence."""
from __future__ import annotations

import json
import pickle
import sys

import numpy as np

import pytest
import torch

from scripts import export_models
from src.common import digest
from src.training import tabular
from src.training.serialization import serialize_tabular_model


def test_neural_export_contains_only_architecture_and_identical_weights(tmp_path, monkeypatch):
    monkeypatch.setattr(export_models, "ARTIFACTS_ROOT", tmp_path / "artifacts")
    source = export_models.ARTIFACTS_ROOT / "runs" / "fixture"
    source.mkdir(parents=True)
    architecture = {
        "mode": "shared", "embedding_dim": 2, "hidden_dim": 4,
        "heads": 1, "layers": 1, "ff_dim": 8, "dropout": 0.1,
        "max_history": 3, "item_count": 5, "token_count": 6,
    }
    weights = {
        "shared.embedding.weight": torch.tensor([[0.0, 1.0], [2.0, 3.0]]),
        "shared.output.bias": torch.tensor([0.25]),
    }
    config = {
        **architecture, "config_hash": "fixture-config", "data_hash": "fixture-data",
        "created_at": "private-date", "output_root": "private-machine-path",
        "provenance": {"code_commit": "private-commit", "hardware": "private-hardware"},
    }
    checkpoint = {
        "models": weights, "config_hash": config["config_hash"], "data_hash": config["data_hash"],
        "optimizer": {"private-optimizer-state": torch.tensor([9.0])},
        "scheduler": {"private-scheduler-step": 100},
        "rng": {"private-random-state": torch.tensor([42], dtype=torch.uint8)},
        "created_at": "private-date", "config": config,
        "extra": {"validation_metrics": {"private-score": 0.123}},
    }
    torch.save(checkpoint, source / "best.pt")
    (source / "config.json").write_text(json.dumps(config), encoding="utf-8")
    source_hash = digest(source / "best.pt")

    result = export_models.export_run("fixture")
    target = export_models.ARTIFACTS_ROOT / "models" / "fixture" / "weights.pt"
    payload = torch.load(target, map_location="cpu", weights_only=True)
    assert set(payload) == {"format_version", "architecture", "state_dict"}
    assert payload["format_version"] == 1
    assert payload["architecture"] == architecture
    assert set(payload["architecture"]) == set(export_models.ARCHITECTURE_FIELDS)
    assert set(payload["state_dict"]) == set(weights)
    for name, tensor in weights.items():
        assert torch.equal(payload["state_dict"][name], tensor)
    assert result["tensor_roundtrip_equal"] is True
    assert result["export_sha256"] == digest(target)
    assert digest(source / "best.pt") == source_hash, "Export must preserve original training evidence"


@pytest.mark.parametrize("run_id", ["..", "../fixture", "fixture/name", "fixture\\name", "C:\\models", "/tmp/model"])
def test_export_rejects_unsafe_run_directory_before_writing(tmp_path, monkeypatch, run_id):
    monkeypatch.setattr(export_models, "ARTIFACTS_ROOT", tmp_path / "artifacts")
    with pytest.raises(ValueError, match="safe directory name"):
        export_models.export_run(run_id)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("model_name,id_only", [("cheap", False), ("logistic", False), ("logistic", True)])
def test_legacy_cli_model_export_loads_normally_and_preserves_predictions(
    tmp_path, monkeypatch, model_name, id_only
):
    monkeypatch.setattr(export_models, "ARTIFACTS_ROOT", tmp_path / "artifacts")
    source = export_models.ARTIFACTS_ROOT / "runs" / "fixture"
    source.mkdir(parents=True)
    record = {
        "task": "search", "split": "train", "item_raw": ["10", "20"],
        "item_ids": [2, 3], "labels": [1, 0], "caption_ids": [[2, 3], [4]],
        "query_ids": [2, 3], "metadata_present": [True, True], "history": [],
        "request_gap": 0,
    }
    model = (tabular.CheapBaseline(id_only=id_only) if model_name == "cheap"
             else tabular.TabularBaseline(id_only=id_only)).fit([record])
    expected = model.predict_request(record)
    # Reproduce the historical -m serialization without altering canonical classes.
    for value in (model, model.extractor):
        legacy_type = type(type(value).__name__, (type(value),), {"__module__": "__main__"})
        monkeypatch.setattr(sys.modules["__main__"], legacy_type.__name__, legacy_type, raising=False)
        value.__class__ = legacy_type
    legacy_payload = pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
    (source / "model.pkl").write_bytes(legacy_payload)
    source_hash = digest(source / "model.pkl")

    # New CLI serialization must also work while its original classes live in __main__.
    stable = pickle.loads(serialize_tabular_model(model))
    assert type(stable).__module__ == "src.training.tabular"
    np.testing.assert_array_equal(stable.predict_request(record), expected)

    result = export_models.export_run("fixture")
    target = export_models.ARTIFACTS_ROOT / "models" / "fixture" / "model.pkl"
    with target.open("rb") as handle:
        restored = pickle.load(handle)
    assert type(restored).__module__ == "src.training.tabular"
    assert type(restored.extractor).__module__ == "src.training.tabular"
    assert restored.extractor.frequency == model.extractor.frequency
    np.testing.assert_array_equal(restored.predict_request(record), expected)
    assert result["canonical_class_modules"] is True
    assert result["fitted_state_roundtrip_equal"] is True
    assert digest(source / "model.pkl") == source_hash
