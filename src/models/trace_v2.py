"""TRACE V2: query-conditioned isolated streams and bounded selected transfer.

Independent implementation using standard Transformer, MLP and ranking losses.
Candidate tensors are flat: no candidate truncation and no B x max-slate padding.
Labels are collated for the loss only; the model never reads them or raw rows.
"""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from src.features.temporal import gap_bucket
from src.models.neural import _mean


def collate_v2(examples, max_history=50):
    if not examples or max_history < 1:
        raise ValueError("Need requests and positive history capacity")
    count = len(examples)
    streams = []
    for opposite in (False, True):
        for row in examples:
            if row["task"] not in ("search", "recommendation"):
                raise ValueError("Unknown task")
            current_bit = 1 if row["task"] == "search" else 2
            bit = 3 - current_bit if opposite else current_bit
            history = row.get("history", [])[-max_history:]
            times = [int(x["timestamp_ms"]) for x in history]
            if any(t >= int(row["timestamp_ms"]) for t in times) or any(a >= b for a, b in zip(times, times[1:])):
                raise ValueError("History must be strictly earlier and chronological")
            streams.append((row, bit, [x for x in history if int(x["service_mask"]) & bit]))
    length = max(1, max(len(history) for _, _, history in streams))
    batch: dict[str, Any] = {"history_mask": torch.zeros(2 * count, length, dtype=torch.bool),
        "history_service": torch.zeros(2 * count, length, dtype=torch.long),
        "history_gap": torch.zeros(2 * count, length, dtype=torch.long),
        "history_age": torch.zeros(2 * count, length, dtype=torch.long),
        "context_scalars": torch.zeros(count, 8),
        "task": torch.tensor([0 if r["task"] == "search" else 1 for r in examples]),
        "query_present": torch.tensor([bool(r.get("query_ids")) for r in examples], dtype=torch.float32),
        "raw": examples}
    ragged = {k: [] for k in ("item_ids", "candidate_owner", "labels", "caption_ids", "caption_owner",
                              "query_ids", "query_owner", "history_rec_ids", "history_rec_owner",
                              "history_query_ids", "history_query_owner", "history_query_block")}
    relevance, caption_present, lengths = [], [], []
    for row_index, example in enumerate(examples):
        items = list(example["item_ids"])
        labels = example.get("labels", [0] * len(items))
        captions = example.get("caption_ids", [[] for _ in items])
        if not items or any(i < 1 for i in items) or len(labels) != len(items) or len(captions) != len(items):
            raise ValueError("Candidates, captions and labels must align; PAD is not a candidate")
        if any(y not in (0, 1) for y in labels):
            raise ValueError("Labels must be binary")
        features = example.get("relevance_features", [[0., 0., 0., 0.] for _ in items])
        if len(features) != len(items) or any(len(x) != 4 for x in features):
            raise ValueError("Four candidate-aligned relevance features required")
        lengths.append(len(items))
        for item, label, caption, feature in zip(items, labels, captions, features):
            owner = len(ragged["item_ids"])
            ids = [int(x) for x in caption[:50] if x != 0]
            if any(x < 1 for x in ids):
                raise ValueError("Negative token ID")
            ragged["item_ids"].append(int(item))
            ragged["candidate_owner"].append(row_index)
            ragged["labels"].append(label)
            ragged["caption_ids"].extend(ids)
            ragged["caption_owner"].extend([owner] * len(ids))
            relevance.append(feature)
            caption_present.append(bool(ids))
        query = [int(x) for x in example.get("query_ids", [])[:50] if x != 0]
        if any(x < 1 for x in query):
            raise ValueError("Negative query ID")
        ragged["query_ids"].extend(query)
        ragged["query_owner"].extend([row_index] * len(query))
    for stream_index, (example, bit, history) in enumerate(streams):
        row_index = stream_index % count
        opposite = stream_index >= count
        n = len(history)
        last_gap = (int(example["timestamp_ms"]) - int(history[-1]["timestamp_ms"])) / 1000 if n else 0
        batch["context_scalars"][row_index, 1 if opposite else 0] = bool(n)
        batch["context_scalars"][row_index, 3 if opposite else 2] = math.log1p(n) / math.log1p(max_history)
        batch["context_scalars"][row_index, 5 if opposite else 4] = math.log1p(min(last_gap, 1e6)) / math.log1p(1e6)
        for position, block in enumerate(history):
            owner = stream_index * length + position
            timestamp = int(block["timestamp_ms"])
            batch["history_mask"][stream_index, position] = True
            batch["history_service"][stream_index, position] = bit
            previous = int(history[position - 1]["timestamp_ms"]) if position else None
            batch["history_gap"][stream_index, position] = gap_bucket(timestamp - previous) if previous is not None else 0
            batch["history_age"][stream_index, position] = gap_bucket(int(example["timestamp_ms"]) - timestamp)
            if bit == 2:
                ids = [int(x) for x in block.get("rec_item_ids", []) if x != 0]
                ragged["history_rec_ids"].extend(ids)
                ragged["history_rec_owner"].extend([owner] * len(ids))
            else:
                for query in block.get("search_query_ids", []):
                    index = len(ragged["history_query_block"])
                    ids = [int(x) for x in query[:50] if x != 0]
                    ragged["history_query_block"].append(owner)
                    ragged["history_query_ids"].extend(ids)
                    ragged["history_query_owner"].extend([index] * len(ids))
    batch["context_scalars"][:, 6:] = F.one_hot(batch["task"], 2).float()
    for name, values in ragged.items():
        batch[name] = torch.tensor(values, dtype=torch.float32 if name == "labels" else torch.long)
    batch["relevance_features"] = torch.tensor(relevance, dtype=torch.float32)
    batch["caption_present"] = torch.tensor(caption_present, dtype=torch.float32)
    batch["lengths"] = lengths
    return batch


def _mlp(input_dim, hidden_dims, dropout):
    layers = []
    for width in hidden_dims:
        layers.extend([nn.Linear(input_dim, width), nn.GELU(), nn.Dropout(dropout)])
        input_dim = width
    layers.append(nn.Linear(input_dim, 1))
    return nn.Sequential(*layers)


class TraceV2(nn.Module):
    def __init__(self, item_count, token_count, embedding_dim=16, hidden_dim=64, heads=2,
                 ff_dim=128, dropout=.1, max_history=50, identity_dropout=.1, residual_bound=2.):
        super().__init__()
        if min(item_count, token_count) < 2 or hidden_dim % heads or not 0 <= identity_dropout <= 1:
            raise ValueError("Invalid model dimensions/dropout")
        self.max_history, self.hidden_dim = max_history, hidden_dim
        self.identity_dropout, self.residual_bound = identity_dropout, residual_bound
        self.item_embedding = nn.Embedding(item_count, embedding_dim, padding_idx=0)
        self.token_embedding = nn.Embedding(token_count, embedding_dim, padding_idx=0)
        self.query_projection = nn.Linear(embedding_dim + 1, hidden_dim)
        self.candidate_projection = nn.Linear(2 * embedding_dim + 2, hidden_dim)
        self.search_projection = nn.Linear(embedding_dim, hidden_dim)
        self.rec_projection = nn.Linear(embedding_dim, hidden_dim)
        self.service_embedding = nn.Embedding(3, hidden_dim, padding_idx=0)
        self.gap_embedding = nn.Embedding(7, hidden_dim)
        self.age_embedding = nn.Embedding(7, hidden_dim)
        self.position_embedding = nn.Embedding(max_history, hidden_dim)
        self.null_query = nn.Parameter(torch.zeros(hidden_dim))
        self.empty_same = nn.Parameter(torch.zeros(hidden_dim))
        layer = nn.TransformerEncoderLayer(hidden_dim, heads, ff_dim, dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        self.history_norm = nn.LayerNorm(hidden_dim)
        self.pool_query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.pool_key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.base = _mlp(6 * hidden_dim + 5, [128, 64], dropout)
        self.residual = _mlp(7 * hidden_dim + 3, [128, 64], dropout)
        self.gate = _mlp(5 * hidden_dim + 8, [64], 0.)
        self.relevance = nn.Linear(4, 1, bias=False)
        nn.init.normal_(self.residual[-1].weight, std=1e-3)
        nn.init.zeros_(self.residual[-1].bias)
        with torch.no_grad():
            self.token_embedding.weight[1].zero_()

    def tokens(self, ids):
        # Unknown tokens contribute zero but count in non-PAD pool denominator.
        return self.token_embedding(ids) * (ids > 1)[:, None]

    def encode_context(self, batch):
        b = len(batch["task"])
        valid = batch["history_mask"]
        n, length = valid.shape
        query = _mean(self.tokens(batch["query_ids"]), batch["query_owner"], b)
        q = self.query_projection(torch.cat([query, batch["query_present"][:, None]], -1))
        q = torch.where((batch["task"] == 0)[:, None], q, self.null_query[None])
        rec = _mean(self.item_embedding(batch["history_rec_ids"]), batch["history_rec_owner"], n * length)
        query_blocks = batch["history_query_block"]
        queries = _mean(self.tokens(batch["history_query_ids"]), batch["history_query_owner"], len(query_blocks))
        search = _mean(queries, query_blocks, n * length)
        rec = self.rec_projection(rec).view(n, length, -1)
        search = self.search_projection(search).view(n, length, -1)
        x = torch.where((batch["history_service"] == 1)[:, :, None], search, rec)
        positions = torch.arange(length, device=x.device)[None].expand(n, -1)
        x = x + self.service_embedding(batch["history_service"]) + self.gap_embedding(batch["history_gap"])
        x = x + self.age_embedding(batch["history_age"]) + self.position_embedding(positions)
        safe_valid = valid.clone()
        empty = ~valid.any(1)
        safe_valid[:, 0] |= empty
        x = torch.where(empty[:, None, None], 0., x)
        mask = torch.ones(length, length, dtype=torch.bool, device=x.device).triu(1)
        h = self.history_norm(self.encoder(x, mask=mask, src_key_padding_mask=~safe_valid, is_causal=True))
        pool_logits = (self.pool_key(h) * self.pool_query(torch.cat([q, q]))[:, None]).sum(-1) / math.sqrt(self.hidden_dim)
        weights = torch.softmax(pool_logits.masked_fill(~safe_valid, float("-inf")), 1)
        pooled = (weights[:, :, None] * h).sum(1)
        same = torch.where(empty[:b, None], self.empty_same[None], pooled[:b])
        cross = torch.where(empty[b:, None], 0., pooled[b:])
        scalars = batch["context_scalars"]
        vg = torch.cat([q, same, cross, q * cross, same * cross, scalars], -1)
        gate_logit = self.gate(vg).squeeze(-1)
        return {"q": q, "same": same, "cross": cross, "vg": vg, "gate_logit": gate_logit,
                "gate": scalars[:, 1] * torch.sigmoid(gate_logit), "scalars": scalars,
                "attention": weights}

    def forward(self, batch, candidate_chunk=256, gate_override=None):
        context = self.encode_context(batch)
        ids = batch["item_ids"]
        known = ids > 1
        if self.training and self.identity_dropout:
            known = known & (torch.rand(ids.shape, device=ids.device) >= self.identity_dropout)
            ids = torch.where(known, ids, torch.ones_like(ids))
        captions = _mean(self.tokens(batch["caption_ids"]), batch["caption_owner"], len(ids))
        candidates = self.candidate_projection(torch.cat([self.item_embedding(ids), captions,
                    known.float()[:, None], batch["caption_present"][:, None]], -1))
        if candidate_chunk < 1:
            raise ValueError("Candidate chunk must be positive")
        owner = batch["candidate_owner"]
        base, residual = [], []
        for start in range(0, len(ids), candidate_chunk):
            sl = slice(start, start + candidate_chunk)
            rows = owner[sl]
            q, same, cross = [context[k][rows] for k in ("q", "same", "cross")]
            c = candidates[sl]
            scalars = context["scalars"][rows]
            base_scalars = scalars[:, [0, 2, 4, 6, 7]]
            base.append(self.base(torch.cat([q, same, c, q * c, same * c, abs(q - c), base_scalars], -1)).squeeze(-1)
                        + self.relevance(batch["relevance_features"][sl]).squeeze(-1))
            features = torch.cat([q, same, cross, c, q * cross, c * cross, same * cross,
                                  scalars[:, [6, 7, 1]]], -1)
            residual.append(scalars[:, 1] * self.residual_bound * torch.tanh(self.residual(features).squeeze(-1)))
        z0, d = torch.cat(base), torch.cat(residual)
        gate = context["gate"] if gate_override is None else context["scalars"][:, 1] * float(gate_override)
        return {**context, "base": z0, "always": z0 + d, "residual": d, "final": z0 + gate[owner] * d}


def group_bce(logits, batch):
    values = F.binary_cross_entropy_with_logits(logits, batch["labels"], reduction="none")
    owners = batch["candidate_owner"]
    sums = values.new_zeros(len(batch["task"])).index_add_(0, owners, values)
    return sums / torch.bincount(owners, minlength=len(sums)).clamp_min(1)


def pairwise_loss(logits, batch, cap=32):
    """Deterministic bounded within-request clicked/unclicked pairs, full BCE retained."""
    losses = []
    offset = 0
    for length in batch["lengths"]:
        labels = batch["labels"][offset:offset + length]
        scores = logits[offset:offset + length]
        positive, negative = torch.where(labels == 1)[0], torch.where(labels == 0)[0]
        if len(positive) and len(negative):
            pairs = min(cap, len(positive) * len(negative))
            # Equally spaced flattened Cartesian indices; deterministic and resume-safe.
            indices = torch.linspace(0, len(positive) * len(negative) - 1, pairs, device=logits.device).long()
            losses.append(F.softplus(-(scores[positive[indices // len(negative)]] - scores[negative[indices % len(negative)]] )).mean())
        offset += length
    # Normalize over all search requests, including degenerate groups with zero
    # pair loss. Microbatch-weighted reduction then equals a complete batch.
    return torch.stack(losses).sum() / len(batch["lengths"]) if losses else logits.sum() * 0


def utility_targets(output, batch, margin=.005):
    delta = (group_bce(output["base"], batch) - group_bce(output["always"], batch)).detach()
    eligible = []
    for row in batch["raw"]:
        labels = row.get("labels", [])
        raw = row.get("item_raw", row["item_ids"])
        eligible.append(row["task"] == "search" and len(set(raw)) >= 2 and 0 < sum(labels) < len(labels))
    eligible = torch.tensor(eligible, dtype=torch.bool, device=delta.device)
    eligible &= (output["scalars"][:, 1] > 0) & torch.isfinite(delta) & (delta.abs() >= margin)
    return delta, (delta >= margin).float(), eligible


def gate_auxiliary(model, output, batch, margin=.005):
    delta, targets, eligible = utility_targets(output, batch, margin)
    # Always touch gate parameters, including the zero-eligible case.
    logits = model.gate(output["vg"].detach()).squeeze(-1)
    loss = F.binary_cross_entropy_with_logits(logits[eligible], targets[eligible]) if eligible.any() else logits.sum() * 0
    return loss, {"eligible": int(eligible.sum()), "helpful": int((targets[eligible] == 1).sum()),
                  "harmful": int((targets[eligible] == 0).sum())}
