"""Meaningful feature, masking, optimization, and continuation contracts."""
from __future__ import annotations

import copy
import dataclasses

import pytest
import torch

from src.models.neural import CompactRanker, collate_requests, supervised_loss
from src.training.checkpoint import load_checkpoint
from src.training.neural import TrainConfig, microbatches, train


def request(task="search", history=None, items=None, labels=None):
    items = [2, 3, 1] if items is None else items
    return {"user_raw": "u1", "request_id": "r1", "item_raw": [str(x) for x in items],
            "item_ids": items, "caption_ids": [[2, 3], [1], []][:len(items)],
            "query_ids": [2, 4] if task == "search" else [],
            "labels": [1, 0, 0][:len(items)] if labels is None else labels,
            "task": task, "timestamp_ms": 2_000_000, "request_gap": 3,
            "history": history or [], "cohort": "transition"}


def history():
    return [
        {"rec_item_ids": [2, 1, 3], "search_query_ids": [[2, 4], [3], []],
         "service_mask": 3, "gap_bucket": 0, "timestamp_ms": 1_000_000},
        {"rec_item_ids": [3], "search_query_ids": [], "service_mask": 2,
         "gap_bucket": 2, "timestamp_ms": 1_200_000},
    ]


def model(history_mode="joint", dropout=0.):
    torch.manual_seed(42)
    torch.set_num_threads(1)
    return CompactRanker(8, 8, embedding_dim=8, hidden_dim=16, heads=2,
                         ff_dim=32, dropout=dropout, history_mode=history_mode).eval()


def test_extra_candidate_and_history_padding_preserves_scores():
    net = model()
    first = request(history=history()[:1], items=[2], labels=[1])
    extra = request(history=history())
    with torch.no_grad():
        solo = net(collate_requests([first]))[0, 0]
        padded = net(collate_requests([first, extra]))[0, 0]
    torch.testing.assert_close(solo, padded, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("history_mode", ["joint", "task_only"])
def test_empty_oov_query_history_are_finite(history_mode):
    net = model(history_mode)
    example = request(items=[1], labels=[0])
    example["caption_ids"], example["query_ids"] = [[]], []
    with torch.no_grad():
        values = net(collate_requests([example, request("recommendation")]))
    assert torch.isfinite(values).all()


def test_permuting_simultaneous_items_queries_tokens_preserves_scores():
    net = model()
    example = request(history=history())
    changed = copy.deepcopy(example)
    for block in changed["history"]:
        block["rec_item_ids"].reverse()
        block["search_query_ids"].reverse()
        for query in block["search_query_ids"]:
            query.reverse()
    with torch.no_grad():
        first = net(collate_requests([example]))
        second = net(collate_requests([changed]))
    torch.testing.assert_close(first, second, atol=1e-6, rtol=1e-6)


def test_changing_current_labels_does_not_change_logits():
    net = model()
    example = request(history=history())
    changed = copy.deepcopy(example)
    changed["labels"] = [0, 1, 1]
    with torch.no_grad():
        first = net(collate_requests([example]))
        second = net(collate_requests([changed]))
    torch.testing.assert_close(first, second, atol=0, rtol=0)


@pytest.mark.parametrize("history_mode", ["joint", "task_only"])
def test_chunked_scoring_and_group_loss_match(history_mode):
    net = model(history_mode)
    batch = collate_requests([request(history=history()), request("recommendation")])
    with torch.no_grad():
        whole, chunks = net(batch), net(batch, candidate_chunk_size=1)
    torch.testing.assert_close(whole, chunks, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(supervised_loss(whole, batch), supervised_loss(chunks, batch))


def test_task_only_ignores_other_service_content_and_timing():
    net = model("task_only")
    example = request(history=history())
    changed = copy.deepcopy(example)
    changed["history"][0]["rec_item_ids"] = [6, 7]
    changed["history"][1]["rec_item_ids"] = [6, 7]
    changed["history"][1]["timestamp_ms"] = 1_800_000
    changed["history"][1]["gap_bucket"] = 4
    changed["request_gap"] = 1
    with torch.no_grad():
        first = net(collate_requests([example]))
        second = net(collate_requests([changed]))
    torch.testing.assert_close(first, second, atol=0, rtol=0)


def test_future_history_rejected():
    example = request(history=history())
    example["history"][-1]["timestamp_ms"] = example["timestamp_ms"]
    with pytest.raises(ValueError, match="precede"):
        collate_requests([example])


def test_encoder_is_causal_between_ordered_blocks():
    net = model()
    captured = []
    hook = net.encoder.register_forward_hook(lambda module, args, output: captured.append(output.detach().clone()))
    try:
        with torch.no_grad():
            net(collate_requests([request(history=history()[:1])]))
            net(collate_requests([request(history=history())]))
    finally:
        hook.remove()
    torch.testing.assert_close(captured[0][:, 0], captured[1][:, 0], atol=1e-6, rtol=1e-6)


def test_task_only_leading_other_service_block_stays_finite():
    net = model("task_only")
    example = request(history=[
        {"rec_item_ids": [2], "search_query_ids": [], "service_mask": 2,
         "gap_bucket": 0, "timestamp_ms": 1_000_000},
        {"rec_item_ids": [], "search_query_ids": [[2]], "service_mask": 1,
         "gap_bucket": 2, "timestamp_ms": 1_200_000}])
    assert torch.isfinite(net(collate_requests([example]))).all()


def test_small_fixture_loss_decreases():
    net = model().train()
    batch = collate_requests([request(), request("recommendation")])
    optimizer = torch.optim.AdamW(net.parameters(), lr=.01)
    first = float(supervised_loss(net(batch), batch).detach())
    for _ in range(40):
        optimizer.zero_grad(set_to_none=True)
        loss = supervised_loss(net(batch), batch)
        loss.backward()
        optimizer.step()
    last = float(supervised_loss(net(batch), batch).detach())
    assert last < .2 * first


def test_variable_size_microbatches_keep_complete_candidates_and_group_mean():
    net = model()
    examples = [request(items=[2], labels=[1]), request(), request(items=[2, 3], labels=[1, 0])]
    parts = list(microbatches(examples, max_padded_candidates=3))
    assert sorted(len(x["item_ids"]) for part in parts for x in part) == [1, 2, 3]
    with torch.no_grad():
        full = collate_requests(examples)
        expected = supervised_loss(net(full), full)
        actual = 0.
        for part in parts:
            batch = collate_requests(part)
            actual += len(part) / len(examples) * supervised_loss(net(batch), batch)
    torch.testing.assert_close(expected, actual, atol=1e-6, rtol=1e-6)


class FixtureDataset:
    item_count = token_count = 8
    data_hash = "fixture-v1"

    def requests(self, split, task=None):
        assert split in ("train", "validation"), "Trainer must not access test."
        examples = [request(history=history()), request("recommendation", history())]
        return [x for x in examples if task is None or x["task"] == task]


def test_cpu_resume_matches_next_optimizer_updates(tmp_path):
    config = TrainConfig(run_id="whole", steps=6, batch_size=2, threads=1,
                         max_seconds=None, validation_every=0, embedding_dim=8,
                         hidden_dim=16, heads=2, ff_dim=32, dropout=.1,
                         output_root=str(tmp_path), record_ledger=False)
    train(FixtureDataset(), config)
    partial = dataclasses.replace(config, run_id="resumed", profile_steps=3)
    train(FixtureDataset(), partial)
    resumed = dataclasses.replace(partial, profile_steps=0, resume="latest")
    train(FixtureDataset(), resumed)
    whole = torch.load(tmp_path / "whole" / "latest.pt", weights_only=False)
    continued = torch.load(tmp_path / "resumed" / "latest.pt", weights_only=False)
    assert whole["step"] == continued["step"] == 6
    assert whole["sampler"] == continued["sampler"]
    for key, value in whole["models"].items():
        torch.testing.assert_close(value, continued["models"][key], atol=0, rtol=0)
    assert torch.equal(whole["rng"]["torch"], continued["rng"]["torch"])


def test_resume_rejects_different_data_before_mutating_model(tmp_path):
    config = TrainConfig(run_id="run", steps=1, batch_size=1, threads=1,
                         max_seconds=None, validation_every=0, embedding_dim=8,
                         hidden_dim=16, ff_dim=32, output_root=str(tmp_path), record_ledger=False)
    train(FixtureDataset(), config)
    dataset = FixtureDataset()
    dataset.data_hash = "different-data"
    with pytest.raises(ValueError, match="dataset hash mismatch"):
        train(dataset, dataclasses.replace(config, resume="latest"))
    with pytest.raises(ValueError, match="run identity mismatch"):
        train(FixtureDataset(), dataclasses.replace(config, run_id="renamed",
              resume=str(tmp_path / "run" / "latest.pt")))


def test_ledger_includes_study_fields_and_actual_training_evidence(tmp_path, monkeypatch):
    import src.training.ledger as ledger
    events = []
    monkeypatch.setattr(ledger, "append_run", events.append)
    monkeypatch.setattr(ledger, "provenance", lambda manifest: {
        "code_commit": None, "code_content_hash": "code-v1", "environment_lock": "environment-v1",
        "raw_archive_sha256": "release-v1", "cohort_hash": "cohort-v1", "split_hash": "split-v1",
        "feature_policy_hash": "features-v1", "vocabulary_hash": "vocab-v1"})
    config = TrainConfig(run_id="ledger-fixture", parent_run="pilot-fixture", steps=1,
                         batch_size=2, threads=1, max_seconds=None, validation_every=0,
                         embedding_dim=8, hidden_dim=16, ff_dim=32, output_root=str(tmp_path))
    train(FixtureDataset(), config)
    required = {"run_id", "parent_run", "question", "code_commit", "environment_lock",
                "data_release_hash", "cohort_hash", "split_hash", "feature_policy_hash", "vocabulary_hash",
                "candidate_policy", "model_config", "seed", "optimizer_updates", "processed_examples",
                "precision", "hardware", "wall_time", "peak_memory", "teacher_hash", "validation_metrics",
                "status", "interpretation"}
    assert len(events) == 2
    assert required <= events[0].keys()
    assert required <= events[1].keys()
    assert events[0]["status"] == "started"
    assert events[1]["optimizer_updates"] == 1
    assert events[1]["processed_examples"]["search_groups"] == 2
    assert events[1]["processed_examples"]["recommendation_exposures"] == 2
    assert events[1]["validation_metrics"]["step"] == 1
    assert events[1]["teacher_hash"] is None
    assert events[1]["parent_run"] == "pilot-fixture"
    assert events[1]["data_release_hash"] == "release-v1"
