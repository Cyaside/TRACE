"""Bounded train-only input and task-gradient diagnosis; no optimizer steps."""
from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path
import time

import numpy as np
import torch

from src.common import read_json, write_json
from src.features.v2_dataset import V2Dataset, V2Sampler
from src.models.trace_v2 import collate_v2, group_bce, pairwise_loss
from src.training.trace_v2 import load_export


def diagnose(dataset, model, output, batches=8, threads=2):
    torch.set_num_threads(threads)
    torch.manual_seed(1042)
    np.random.seed(1042)
    sampler = V2Sampler(dataset, 1042)
    model.eval()
    # Exclude giant ID tables from stored gradients. The diagnostic explicitly
    # concerns shared representation/scorer parameters, not the embedding table.
    parameters = [(n,p) for n,p in model.named_parameters() if "embedding" not in n]
    records = []
    started = time.perf_counter()
    for index in range(batches):
        gradients, losses, probes = {}, {}, {}
        for task in ("search", "recommendation"):
            examples = sampler.batch(task, 8 if task == "search" else 64)
            batch = collate_v2(examples)
            out = model(batch)
            loss = group_bce(out["final"], batch).mean()
            if task == "search":
                loss = loss + .1*pairwise_loss(out["final"], batch)
                with torch.no_grad():
                    probes["original_bce"] = float(group_bce(out["final"], batch).mean())
                    for name in ("query", "lexical", "caption"):
                        masked = copy.deepcopy(examples)
                        for row in masked:
                            if name == "query":
                                row["query_ids"] = []
                                row["relevance_features"] = [[0.,0.,0.,f[3]] for f in row["relevance_features"]]
                            elif name == "lexical":
                                row["relevance_features"] = [[0.,0.,0.,f[3]] for f in row["relevance_features"]]
                            else:
                                row["caption_ids"] = [[] for _ in row["item_ids"]]
                                row["relevance_features"] = [[0.,0.,0.,f[3]] for f in row["relevance_features"]]
                        changed = collate_v2(masked)
                        changed_out = model(changed)
                        probes[f"without_{name}_bce"] = float(group_bce(changed_out["final"], changed).mean())
                        probes[f"without_{name}_mean_abs_logit_change"] = float((out["final"]-changed_out["final"]).abs().mean())
            values = torch.autograd.grad(loss, [p for _,p in parameters], allow_unused=True)
            gradients[task] = [torch.zeros_like(p) if g is None else g.detach() for (_,p),g in zip(parameters, values)]
            losses[task] = float(loss.detach())
        dot = sum(float((a*b).sum()) for a,b in zip(gradients["search"], gradients["recommendation"]))
        norms = {task: math.sqrt(sum(float((g*g).sum()) for g in grads)) for task,grads in gradients.items()}
        cosine = dot/max(1e-12,norms["search"]*norms["recommendation"])
        records.append({"batch": index, "cosine": cosine, "norms": norms, "loss": losses, "frozen_input_masks": probes})
    result = {"data_hash": dataset.data_hash, "split": "train", "optimizer_updates": 0,
        "batches": batches, "search_groups_per_batch": 8, "rec_exposures_per_batch": 64,
        "gradient_scope": "all non-embedding model parameters; excludes ID/token/service/time/position embedding tables",
        "objective": "final-path BCE plus existing search pairwise term; probes/utility auxiliary excluded",
        "mean_cosine": float(np.mean([r["cosine"] for r in records])),
        "negative_cosine_fraction": float(np.mean([r["cosine"] < 0 for r in records])),
        "seconds": time.perf_counter()-started, "records": records,
        "interpretation": "Exploratory small train-only diagnostic, not a causal demonstration or retrained input ablation.",
        "test_accessed": False}
    write_json(Path(output),result)
    print({k:result[k] for k in ("batches","mean_cosine","negative_cosine_fraction","seconds")},flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dataset = V2Dataset(args.data)
    model = load_export(args.weights)
    diagnose(dataset,model,args.output)
