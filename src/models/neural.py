"""Small exposure-ranking encoders with ordered, unordered-within-time histories.

This is an independently implemented SASRec-derived adaptation: simultaneous
exposure blocks are pooled and the target is logged-exposure BCE, not SASRec's
original next-click objective (https://arxiv.org/abs/1808.09781).
"""
from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import Tensor, nn

GAP_THRESHOLDS_MS = (60_000, 300_000, 1_800_000, 7_200_000, 86_400_000)


def _ids(values):
    values = [int(x) for x in values]
    if any(x < 0 for x in values):
        raise ValueError("Encoded identities must be nonnegative; PAD=0, OOV=1.")
    return [x for x in values if x != 0]


def collate_requests(examples: Sequence[dict[str, Any]], max_history: int = 50) -> dict[str, Any]:
    """Collate ragged identities without truncating candidates or block contents.

    Histories retain the last ``max_history`` *blocks*. Ragged token/identity
    owners avoid padding every simultaneous block to its largest item/query
    count. Labels are carried only for the loss and never read by the model.
    """
    if not examples or max_history < 1:
        raise ValueError("Need a nonempty batch and positive max_history.")
    b = len(examples)
    c = max(len(x["item_ids"]) for x in examples)
    if c == 0 or any(not x["item_ids"] for x in examples):
        raise ValueError("Every ranking request needs at least one candidate.")
    histories = [x.get("history", [])[-max_history:] for x in examples]
    h = max(1, max(len(x) for x in histories))
    batch: dict[str, Any] = {
        "item_ids": torch.zeros((b, c), dtype=torch.long),
        "candidate_mask": torch.zeros((b, c), dtype=torch.bool),
        "labels": torch.zeros((b, c), dtype=torch.float32),
        "history_mask": torch.zeros((b, h), dtype=torch.bool),
        "service_mask": torch.zeros((b, h), dtype=torch.long),
        "history_gap": torch.zeros((b, h), dtype=torch.long),
        "history_timestamp": torch.full((b, h), -1, dtype=torch.long),
        "request_timestamp": torch.tensor([int(x.get("timestamp_ms", -1)) for x in examples]),
        "request_gap": torch.tensor([int(x.get("request_gap", 0)) for x in examples]),
        "task": torch.tensor([0 if x["task"] == "search" else 1 for x in examples]),
        "raw": list(examples),
    }
    ragged = {name: [] for name in (
        "caption_ids", "caption_owner", "query_ids", "query_owner",
        "history_rec_ids", "history_rec_owner", "history_query_ids",
        "history_query_owner", "history_query_block",
    )}
    for row, (example, history) in enumerate(zip(examples, histories)):
        if example["task"] not in ("search", "recommendation"):
            raise ValueError("Unknown task.")
        items = [int(x) for x in example["item_ids"]]
        if any(x < 1 for x in items):
            raise ValueError("Actual candidates cannot be PAD or negative.")
        labels = example.get("labels", [0] * len(items))
        if len(labels) != len(items) or any(x not in (0, 1) for x in labels):
            raise ValueError("Candidate-aligned binary labels are required.")
        captions = example.get("caption_ids", [[] for _ in items])
        if len(captions) != len(items):
            raise ValueError("Captions must align with candidate identities.")
        batch["item_ids"][row, :len(items)] = torch.tensor(items)
        batch["candidate_mask"][row, :len(items)] = True
        batch["labels"][row, :len(items)] = torch.tensor(labels, dtype=torch.float32)
        for column, caption in enumerate(captions):
            ids = _ids(caption)
            ragged["caption_ids"].extend(ids)
            ragged["caption_owner"].extend([row * c + column] * len(ids))
        ids = _ids(example.get("query_ids", []))
        ragged["query_ids"].extend(ids)
        ragged["query_owner"].extend([row] * len(ids))
        previous_time = None
        for block_index, block in enumerate(history):
            owner = row * h + block_index
            service = int(block["service_mask"])
            if service not in (1, 2, 3):
                raise ValueError("History blocks must contain search, recommendation, or both.")
            timestamp = int(block.get("timestamp_ms", -1))
            if timestamp >= 0:
                if previous_time is not None and timestamp <= previous_time:
                    raise ValueError("History blocks must be strictly chronological.")
                request_time = int(example.get("timestamp_ms", -1))
                if request_time >= 0 and timestamp >= request_time:
                    raise ValueError("History must precede the current request.")
                previous_time = timestamp
            batch["history_mask"][row, block_index] = True
            batch["service_mask"][row, block_index] = service
            batch["history_gap"][row, block_index] = int(block.get("gap_bucket", 0))
            batch["history_timestamp"][row, block_index] = timestamp
            ids = _ids(block.get("rec_item_ids", []))
            ragged["history_rec_ids"].extend(ids)
            ragged["history_rec_owner"].extend([owner] * len(ids))
            for query in block.get("search_query_ids", []):
                query_index = len(ragged["history_query_block"])
                ragged["history_query_block"].append(owner)
                ids = _ids(query)
                ragged["history_query_ids"].extend(ids)
                ragged["history_query_owner"].extend([query_index] * len(ids))
    for name, values in ragged.items():
        batch[name] = torch.tensor(values, dtype=torch.long)
    return batch


def to_device(batch: dict[str, Any], device: str | torch.device) -> dict[str, Any]:
    return {key: value.to(device) if isinstance(value, Tensor) else value for key, value in batch.items()}


def _mean(values: Tensor, owners: Tensor, count: int, empty: Tensor | None = None) -> Tensor:
    result = values.new_zeros((count, values.shape[-1]))
    denominator = values.new_zeros((count, 1))
    if owners.numel():
        result.index_add_(0, owners, values)
        denominator.index_add_(0, owners, values.new_ones((len(owners), 1)))
    result = result / denominator.clamp_min(1)
    if empty is not None:
        result = torch.where(denominator > 0, result, empty[None, :])
    return result


class CompactRanker(nn.Module):
    """Joint or task-specific block Transformer plus candidate scoring MLP."""

    def __init__(self, item_count: int, token_count: int, embedding_dim: int = 32,
                 hidden_dim: int = 64, heads: int = 2, layers: int = 1,
                 ff_dim: int = 128, dropout: float = .1, max_history: int = 50,
                 history_mode: str = "joint", gap_count: int = 8):
        super().__init__()
        if item_count < 2 or token_count < 2 or max_history < 1:
            raise ValueError("Vocabulary must reserve PAD/OOV and history capacity must be positive.")
        if history_mode not in ("joint", "task_only"):
            raise ValueError("history_mode must be joint or task_only.")
        self.max_history, self.history_mode = max_history, history_mode
        self.item_embedding = nn.Embedding(item_count, embedding_dim, padding_idx=0)
        self.token_embedding = nn.Embedding(token_count, embedding_dim, padding_idx=0)
        self.null_query = nn.Parameter(torch.zeros(embedding_dim))
        self.empty_history = nn.Parameter(torch.zeros(hidden_dim))
        self.history_projection = nn.Linear(2 * embedding_dim, hidden_dim)
        self.query_projection = nn.Linear(embedding_dim, hidden_dim)
        self.candidate_projection = nn.Linear(2 * embedding_dim, hidden_dim)
        self.service_embedding = nn.Embedding(4, hidden_dim, padding_idx=0)
        self.gap_embedding = nn.Embedding(gap_count, hidden_dim)
        self.position_embedding = nn.Embedding(max_history, hidden_dim)
        self.task_embedding = nn.Embedding(2, hidden_dim)
        layer = nn.TransformerEncoderLayer(hidden_dim, heads, ff_dim, dropout,
                                           activation="gelu", batch_first=True,
                                           norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.history_norm = nn.LayerNorm(hidden_dim)
        self.context_projection = nn.Sequential(nn.Linear(4 * hidden_dim, hidden_dim),
                                                nn.GELU(), nn.LayerNorm(hidden_dim))
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(3 * hidden_dim, hidden_dim), nn.GELU(),
                          nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
            for _ in range(2)
        ])

    def encode_context(self, batch: dict[str, Any]) -> Tensor:
        valid = batch["history_mask"].clone()
        b, h = valid.shape
        if h > self.max_history:
            raise ValueError("Collate with this model's maximum history length.")
        rec = _mean(self.item_embedding(batch["history_rec_ids"]),
                    batch["history_rec_owner"], b * h).view(b, h, -1)
        query_blocks = batch["history_query_block"]
        historical_queries = _mean(self.token_embedding(batch["history_query_ids"]),
                                   batch["history_query_owner"], len(query_blocks), self.null_query)
        search = _mean(historical_queries, query_blocks, b * h).view(b, h, -1)
        service = batch["service_mask"]
        gaps, request_gap = batch["history_gap"], batch["request_gap"]
        if self.history_mode == "task_only":
            bit = torch.where(batch["task"] == 0, 1, 2)[:, None]
            valid = valid & ((service & bit) != 0)
            service = torch.where(valid, bit, 0)
            rec = rec * (batch["task"] == 1)[:, None, None]
            search = search * (batch["task"] == 0)[:, None, None]
            if torch.any(valid & (batch["history_timestamp"] < 0)) or torch.any(batch["request_timestamp"] < 0):
                raise ValueError("Task-only gaps require exact block/request timestamps.")
            times = torch.where(valid, batch["history_timestamp"], 0)
            latest = times.cummax(dim=1).values
            previous = torch.cat([torch.zeros_like(latest[:, :1]), latest[:, :-1]], dim=1)
            thresholds = times.new_tensor(GAP_THRESHOLDS_MS)
            gaps = torch.where(valid & (previous > 0),
                               torch.bucketize((times - previous).contiguous(), thresholds) + 1, 0)
            request_gap = torch.where(latest[:, -1] > 0,
                torch.bucketize((batch["request_timestamp"] - latest[:, -1]).contiguous(), thresholds) + 1, 0)
            # Pack retained blocks left. Leading masked queries under causal
            # attention otherwise have no legal keys and can produce NaNs.
            order = ((~valid).long() * h + torch.arange(h, device=valid.device)[None, :]).argsort(dim=1)
            rec = rec.gather(1, order[:, :, None].expand_as(rec))
            search = search.gather(1, order[:, :, None].expand_as(search))
            service, gaps = service.gather(1, order), gaps.gather(1, order)
            valid = valid.gather(1, order)
        positions = (valid.long().cumsum(dim=1) - 1).clamp_min(0)
        history = self.history_projection(torch.cat([rec, search], dim=-1))
        history = history + self.service_embedding(service) + self.gap_embedding(gaps)
        history = history + self.position_embedding(positions)
        empty = ~valid.any(dim=1)
        history = torch.where((empty[:, None] & (torch.arange(h, device=valid.device)[None, :] == 0))[:, :, None],
                              self.empty_history[None, None, :], history)
        valid[:, 0] |= empty
        causal = torch.ones((h, h), dtype=torch.bool, device=valid.device).triu(diagonal=1)
        encoded = self.history_norm(self.encoder(history, mask=causal,
                                                 src_key_padding_mask=~valid, is_causal=True))
        pooled = torch.where(valid[:, :, None], encoded, 0).sum(dim=1) / valid.sum(dim=1).clamp_min(1)[:, None]
        current_query = _mean(self.token_embedding(batch["query_ids"]), batch["query_owner"], b, self.null_query)
        current_query = torch.where((batch["task"] == 0)[:, None], current_query, self.null_query[None, :])
        context = torch.cat([pooled, self.query_projection(current_query),
                             self.gap_embedding(request_gap), self.task_embedding(batch["task"])], dim=-1)
        return self.context_projection(context)

    def forward(self, batch: dict[str, Any], candidate_chunk_size: int | None = None) -> Tensor:
        context = self.encode_context(batch)
        b, c = batch["item_ids"].shape
        captions = _mean(self.token_embedding(batch["caption_ids"]), batch["caption_owner"], b * c).view(b, c, -1)
        items = self.item_embedding(batch["item_ids"])
        candidates = self.candidate_projection(torch.cat([items, captions], dim=-1))
        chunk = c if candidate_chunk_size is None else int(candidate_chunk_size)
        if chunk < 1:
            raise ValueError("Candidate chunk size must be positive.")
        outputs = []
        for start in range(0, c, chunk):
            candidate = candidates[:, start:start + chunk]
            expanded = context[:, None, :].expand(-1, candidate.shape[1], -1)
            features = torch.cat([expanded, candidate, expanded * candidate], dim=-1)
            if torch.all(batch["task"] == batch["task"][0]):
                outputs.append(self.heads[int(batch["task"][0])](features).squeeze(-1))
            else:
                task_scores = torch.stack([head(features).squeeze(-1) for head in self.heads], dim=-1)
                outputs.append(task_scores.gather(-1, batch["task"][:, None, None].expand(-1, candidate.shape[1], 1)).squeeze(-1))
        return torch.cat(outputs, dim=1)


def supervised_loss(logits: Tensor, batch: dict[str, Any], reduction: str = "group") -> Tensor:
    """Search group-mean BCE or recommendation exposure-mean BCE."""
    values = nn.functional.binary_cross_entropy_with_logits(logits, batch["labels"], reduction="none")
    mask = batch["candidate_mask"]
    if reduction == "group":
        return ((values * mask).sum(1) / mask.sum(1).clamp_min(1)).mean()
    if reduction == "exposure":
        return values[mask].mean()
    raise ValueError("Unknown supervised reduction.")
