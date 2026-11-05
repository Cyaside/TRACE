"""Matched evaluation contracts, not real-data quality tests."""
import json

import numpy as np
import pytest

from src.common import read_json
from src.evaluation.v2_study import (baselines, branch_diagnostics, extractor_for, feature_rows, FastTabularFeatures,
                                    jsonl, prepare_features, score_array)
from src.features.v2_dataset import V2Dataset
from test_v2_dataset import prepared


def test_disk_features_match_original_extractor_and_task_weights(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    dataset = V2Dataset(prepared(data_root))
    output = tmp_path / "features"
    manifest = prepare_features(dataset, output)
    x = np.load(output / "train-x.npy")
    w = np.load(output / "train-w.npy")
    assert manifest["contract"]["test_accessed"] is False
    assert len(x) == 4 and x.shape[1] == 17
    np.testing.assert_allclose(w[:2].sum(), w[2:].sum())
    np.testing.assert_allclose(w.mean(), 1)
    row = dataset.encode(dataset.indexes[("train", "search")][0])
    extractor = extractor_for(dataset)
    np.testing.assert_allclose(x[:2], feature_rows(extractor, row, dataset))
    raw = dataset.rows.slice(int(dataset.indexes[("train", "search")][0]), 1).to_pylist()[0]
    np.testing.assert_allclose(FastTabularFeatures(dataset)(raw), feature_rows(extractor, row, dataset))
    changed = {**row, "labels": [0, 1]}
    np.testing.assert_array_equal(feature_rows(extractor, row, dataset), feature_rows(extractor, changed, dataset))
    assert manifest == prepare_features(dataset, output)
    with (output / "train-x.npy").open("ab") as handle:
        handle.write(b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        prepare_features(dataset, output)


def test_logistic_and_rule_exports_match_validation_contract(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    dataset = V2Dataset(prepared(data_root))
    output = tmp_path / "baselines"
    results = baselines(dataset, output, threads=1)
    assert len(results) == 3
    for name, result in results.items():
        records = list(jsonl(output / name / "validation.jsonl"))
        assert len(records) == 3
        assert records[0]["item_raw"] == [99, 20]
        assert [r["label"] for r in records[1:]] == [1, 0]
        assert result["test_accessed"] is False
        if name.startswith("logistic"):
            assert result["converged"]
            value = read_json(output / name / "model.json")
            vx = np.load(output / "features/validation-x.npy")[:, value["columns"]]
            logits = ((vx-value["mean"])/value["scale"]) @ np.asarray(value["coef"])[0] + value["intercept"][0]
            saved = records[0]["scores"] + [r["score"] for r in records[1:]]
            np.testing.assert_allclose(logits, saved, atol=1e-12)


def test_frozen_branches_validate_saved_ids_labels_and_logits(tmp_path):
    import torch
    from src.models.trace_v2 import TraceV2
    from src.training.trace_v2 import score_rows
    data_root = tmp_path / "data"
    data_root.mkdir()
    dataset = V2Dataset(prepared(data_root))
    config = {"model": {"max_history": 50}, "recommendation_microbatch": 64,
              "microbatch_requests": 8, "candidate_budget": 2048, "candidate_chunk": 256}
    torch.manual_seed(42)
    model = TraceV2(dataset.item_count, dataset.token_count, dropout=0).eval()
    records = list(score_rows(model, dataset, config, "cpu"))
    saved = tmp_path / "saved.jsonl"
    saved.write_text("".join(json.dumps(r)+"\n" for r in records))
    result = branch_diagnostics(model, dataset, config, saved, tmp_path / "branches", threads=1)
    assert result["parity"]["passed"] and result["parity"]["candidates"] == 4
    assert result["metrics"]["final"] == result["saved_gpu_metrics"]
    records[0]["scores"][0] += 1
    saved.write_text("".join(json.dumps(r)+"\n" for r in records))
    with pytest.raises(AssertionError):
        branch_diagnostics(model, dataset, config, saved, tmp_path / "bad-logits", threads=1)
    records[0]["labels"].reverse()
    saved.write_text("".join(json.dumps(r)+"\n" for r in records))
    with pytest.raises(ValueError, match="alignment"):
        branch_diagnostics(model, dataset, config, saved, tmp_path / "bad-labels", threads=1)
