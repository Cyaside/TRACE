"""Top-k delta-NDCG weighted pair loss for complete binary-label slates.

Inspired by LambdaRank; normalized selected-pair weights are an explicit local
variant, not a claim to reproduce the full LambdaRank/LambdaLoss algorithm.
"""
from __future__ import annotations

import math
import torch
from torch.nn import functional as F


def delta_ndcg_pairwise(logits, batch, cap=32, k=10):
    if cap < 1 or k < 1:
        raise ValueError("Positive pair cap and ranking cutoff required")
    losses, offset = [], 0
    for row, length in zip(batch["raw"], batch["lengths"]):
        scores = logits[offset:offset+length]
        labels = batch["labels"][offset:offset+length]
        positive, negative = torch.where(labels == 1)[0], torch.where(labels == 0)[0]
        if len(positive) and len(negative):
            pairs = min(cap, len(positive)*len(negative))
            flat = torch.linspace(0, len(positive)*len(negative)-1, pairs, device=logits.device).long()
            p, n = positive[flat//len(negative)], negative[flat%len(negative)]
            with torch.no_grad():
                # Same raw-ID tie rule as the evaluator; rankings/weights are
                # constants for autograd, including near-zero equal scores.
                raw = row.get("item_raw", row["item_ids"])
                detached = scores.detach().cpu().tolist()
                order = sorted(range(length), key=lambda i: (-detached[i], int(raw[i])))
                ranks = torch.empty(length, dtype=torch.long, device=logits.device)
                ranks[torch.tensor(order, device=logits.device)] = torch.arange(length, device=logits.device)
                discount = torch.where(ranks < k, 1/torch.log2(ranks.float()+2), 0.)
                ideal = sum(1/math.log2(i+2) for i in range(min(k, len(positive))))
                weights = (discount[p]-discount[n]).abs()/ideal
            losses.append((weights*F.softplus(-(scores[p]-scores[n]))).sum()/weights.sum().clamp_min(1e-12))
        offset += length
    # Degenerate requests contribute zero, preserving microbatch equivalence.
    return torch.stack(losses).sum()/len(batch["lengths"]) if losses else logits.sum()*0
