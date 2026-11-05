"""V2 leakage, reduction, cold-start, gradient routing and resume contracts."""
import copy
import json

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from src.models.trace_v2 import TraceV2, collate_v2, gate_auxiliary, group_bce, pairwise_loss, utility_targets
from src.training.trace_v2 import export_weights, load_export, partition, train


def example(task="search", history=True):
    return {"user_raw": "u1", "request_id": "r1", "item_raw": [20, 30, 99],
            "item_ids": [2, 3, 1], "labels": [1, 0, 0], "caption_ids": [[2, 4], [3], []],
            "query_ids": [2, 4] if task == "search" else [], "task": task,
            "timestamp_ms": 2000000, "cohort": "transition", "relevance_features": [[.5, .5, .3, .8], [0, 0, 0, .2], [0, 0, 0, 0]],
            "history": [{"timestamp_ms": 1000000, "service_mask": 3, "rec_item_ids": [2, 3],
                         "search_query_ids": [[2, 4], [3]]},
                        {"timestamp_ms": 1200000, "service_mask": 2, "rec_item_ids": [3],
                         "search_query_ids": []}] if history else []}


def model(dropout=0., identity_dropout=0.):
    torch.set_num_threads(1)
    torch.manual_seed(42)
    return TraceV2(8, 8, embedding_dim=8, hidden_dim=16, heads=2, ff_dim=32,
                   dropout=dropout, identity_dropout=identity_dropout).eval()


def test_base_isolated_and_no_cross_exact():
    net = model()
    first = example()
    changed = copy.deepcopy(first)
    changed["history"][0]["rec_item_ids"] = [5, 6, 7]
    changed["history"][1]["rec_item_ids"] = [6]
    changed["history"][1]["timestamp_ms"] = 1300000
    with torch.no_grad():
        a, b = net(collate_v2([first])), net(collate_v2([changed]))
    torch.testing.assert_close(a["base"], b["base"], atol=0, rtol=0)
    assert not torch.allclose(a["always"], b["always"], atol=1e-8)
    first["history"] = [{"timestamp_ms": 1000000, "service_mask": 1, "rec_item_ids": [], "search_query_ids": [[2, 4]]}]
    out = net(collate_v2([first]))
    assert torch.equal(out["base"], out["final"])
    assert torch.equal(out["base"], out["always"])
    assert torch.equal(out["gate"], torch.zeros(1))
    assert torch.all(a["residual"].abs() <= 2)


def test_padding_empty_oov_and_labels_do_not_affect_scores():
    net = model()
    first = example(history=False)
    first["query_ids"] = [1]
    extra = example()
    with torch.no_grad():
        single = net(collate_v2([first]))["final"]
        mixed = net(collate_v2([first, extra]))["final"][:3]
    assert torch.isfinite(single).all()
    torch.testing.assert_close(single, mixed, atol=1e-6, rtol=1e-6)
    mutated = copy.deepcopy(first)
    mutated["labels"] = [0, 1, 1]
    mutated["click"] = "forbidden-field-is-ignored"
    torch.testing.assert_close(single, net(collate_v2([mutated]))["final"], atol=0, rtol=0)
    for field in ("labels", "raw"):
        batch = collate_v2([first])
        del batch[field]
        torch.testing.assert_close(single, net(batch)["final"], atol=0, rtol=0)


def test_simultaneous_order_and_strict_timestamp():
    net = model()
    row = example()
    changed = copy.deepcopy(row)
    changed["history"][0]["rec_item_ids"].reverse()
    changed["history"][0]["search_query_ids"].reverse()
    for q in changed["history"][0]["search_query_ids"]:
        q.reverse()
    torch.testing.assert_close(net(collate_v2([row]))["final"], net(collate_v2([changed]))["final"], atol=1e-6, rtol=1e-6)
    row["history"][0]["timestamp_ms"] = row["timestamp_ms"]
    with pytest.raises(ValueError, match="earlier"):
        collate_v2([row])


def test_query_gradient_and_trained_unknown_identity():
    net = model(identity_dropout=1.).train()
    batch = collate_v2([example()])
    loss = group_bce(net(batch)["final"], batch).mean()
    loss.backward()
    assert net.item_embedding.weight.grad[1].abs().sum() > 0
    assert net.query_projection.weight.grad.abs().sum() > 0
    assert net.pool_query.weight.grad.abs().sum() > 0
    assert net.token_embedding.weight.grad[1].abs().sum() == 0


def test_flat_candidates_chunk_and_microbatch_loss_gradient_parity():
    net = model().train()
    rows = [example(), example("recommendation", history=False)]
    batch = collate_v2(rows)
    torch.testing.assert_close(net(batch, 1)["final"], net(batch, 100)["final"], atol=1e-6, rtol=1e-6)
    expected = group_bce(net(batch)["final"], batch).mean()
    expected.backward()
    gradients = {k: p.grad.clone() for k, p in net.named_parameters() if p.grad is not None}
    net.zero_grad()
    actual = 0.
    for part in partition(rows, requests=1, candidates=1):
        b = collate_v2(part)
        loss = group_bce(net(b)["final"], b).mean() * len(part) / len(rows)
        actual += float(loss.detach())
        loss.backward()
    assert actual == pytest.approx(float(expected.detach()), abs=1e-6)
    for k, p in net.named_parameters():
        if k in gradients:
            torch.testing.assert_close(gradients[k], p.grad, atol=2e-6, rtol=1e-5)


def test_losses_utility_margin_and_gate_only_gradient():
    net = model()
    batch = collate_v2([example()])
    output = net(batch)
    bce = F.binary_cross_entropy_with_logits(output["final"], batch["labels"])
    torch.testing.assert_close(group_bce(output["final"], batch).mean(), bce)
    logits = torch.tensor([2., -1., 0.], requires_grad=True)
    expected = F.softplus(torch.tensor([-3., -2.])).mean()
    torch.testing.assert_close(pairwise_loss(logits, batch), expected)
    fabricated = {**output, "base": torch.zeros(3), "always": torch.tensor([4., -4., -4.])}
    delta, target, eligible = utility_targets(fabricated, batch)
    assert delta.item() > .005 and target.item() == 1 and eligible.item()
    loss, stats = gate_auxiliary(net, fabricated, batch)
    loss.backward()
    assert stats["eligible"] == 1
    for name, parameter in net.named_parameters():
        assert (parameter.grad is not None) == name.startswith("gate.")
    net.zero_grad()
    fabricated["always"] = fabricated["base"]
    loss, stats = gate_auxiliary(net, fabricated, batch)
    assert loss.item() == 0 and stats["eligible"] == 0
    loss.backward()


class FixtureDataset:
    data_hash = "fixture-v2"
    item_count = token_count = 8
    indexes = {("train", "search"): np.array([0]), ("train", "recommendation"): np.array([1])}
    rec_offsets = np.array([3])
    rec_count = 3

    def encode(self, index, column=None):
        row = example("search" if index == 0 else "recommendation")
        if column is not None:
            for key in ("item_ids", "item_raw", "labels", "caption_ids", "relevance_features"):
                row[key] = [row[key][column]]
        return row

    def iter_requests(self, split, task):
        assert split == "validation", "Test is forbidden during training"
        yield self.encode(0 if task == "search" else 1)

    def feature_state(self):
        return {"fixture": True}


def config(run_id):
    from pathlib import Path
    value = json.loads((Path(__file__).parents[1] / "configs/trace-v2-kaggle.json").read_text())
    value.update(run_id=run_id, device="cpu", threads=1, steps=6, probe_warmup_steps=2,
                 search_batch=2, recommendation_batch=2, validation_every=3, checkpoint_every=3)
    value["model"].update(embedding_dim=8, hidden_dim=16, ff_dim=32)
    return value


def test_resume_export_and_provenance(tmp_path):
    dataset = FixtureDataset()
    train(dataset, config("whole"), tmp_path)
    train(dataset, config("resumed"), tmp_path, stop_after=3)
    checkpoint = tmp_path / "resumed/latest.pt"
    train(dataset, config("resumed"), tmp_path, resume=checkpoint)
    a = torch.load(tmp_path / "whole/latest.pt", weights_only=False)
    b = torch.load(checkpoint, weights_only=False)
    assert a["step"] == b["step"] == 6
    assert a["sampler"] == b["sampler"]
    for name, value in a["models"].items():
        torch.testing.assert_close(value, b["models"][name], atol=0, rtol=0)
    exported = load_export(tmp_path / "resumed/model-latest.pt")
    source = model(dropout=.1, identity_dropout=.1)
    source.load_state_dict(b["models"])
    batch = collate_v2([example()])
    torch.testing.assert_close(source(batch)["final"], exported(batch)["final"], atol=0, rtol=0)
    payload = torch.load(tmp_path / "resumed/model-latest.pt", weights_only=True)
    assert not {"optimizer", "rng", "sampler", "history"} & payload.keys()
    invalid = config("resumed")
    invalid["model"]["identity_dropout"] = .2
    with pytest.raises(ValueError, match="configuration hash"):
        train(dataset, invalid, tmp_path, resume=checkpoint)


def test_tiny_fixture_can_learn():
    net = model().train()
    batch = collate_v2([example(), example("recommendation")])
    optimizer = torch.optim.AdamW(net.parameters(), lr=.01)
    first = float(group_bce(net(batch)["final"], batch).mean().detach())
    for _ in range(35):
        optimizer.zero_grad()
        loss = group_bce(net(batch)["final"], batch).mean()
        loss.backward()
        optimizer.step()
    last = float(group_bce(net(batch)["final"], batch).mean().detach())
    assert last < first * .25


def test_gpu_replica_gradient_sum_matches_single_model():
    # CPU mathematical fixture verifies the declared replica reduction; actual
    # CUDA allocation/throughput and cross-device transfer are checked in Kaggle.
    from src.training.trace_v2 import task_backward
    cfg = config("replica-check")
    cfg["model"].update(dropout=0., identity_dropout=0.)
    primary = TraceV2(8, 8, **cfg["model"])
    reference, secondary = copy.deepcopy(primary), copy.deepcopy(primary)
    batches = {task: [collate_v2([example(task)])] for task in ("search", "recommendation")}
    task_backward(reference, batches["search"], "search", cfg, 3)
    task_backward(reference, batches["recommendation"], "recommendation", cfg, 3)
    task_backward(primary, batches["search"], "search", cfg, 3)
    task_backward(secondary, batches["recommendation"], "recommendation", cfg, 3)
    for a, b, expected in zip(primary.parameters(), secondary.parameters(), reference.parameters()):
        grad = (torch.zeros_like(a) if a.grad is None else a.grad) + (torch.zeros_like(b) if b.grad is None else b.grad)
        if expected.grad is not None:
            torch.testing.assert_close(grad, expected.grad, atol=1e-6, rtol=1e-5)
