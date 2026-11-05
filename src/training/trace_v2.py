"""Bounded V2 training with automatic profiling and exact continuation state.

Kaggle's two GPUs process paired search/rec tasks using persistent replicas.
Gradients are summed on the primary device before one AdamW step; updated
parameters are copied to the other replica. This is task parallelism, not pooled
VRAM or automatic DDP. No test scoring, sweep, or polling service is started.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import psutil
import torch
from torch.nn import functional as F

from src.common import object_hash, read_json, write_json
from src.features.v2_dataset import V2Dataset, V2Sampler
from src.models.neural import to_device
from src.models.trace_v2 import TraceV2, collate_v2, group_bce, pairwise_loss, utility_targets
from src.training.checkpoint import load_checkpoint, save_checkpoint


def partition(examples, requests=8, candidates=2048):
    """Complete requests retained; flat candidate count bounds each microbatch."""
    current, size = [], 0
    for example in examples:
        n = len(example["item_ids"])
        if current and (len(current) >= requests or size + n > candidates):
            yield current
            current, size = [], 0
        current.append(example)
        size += n
    if current:
        yield current


def _batches(examples, config, device):
    request_limit = (config.get("recommendation_microbatch", 64) if examples[0]["task"] == "recommendation"
                     else config["microbatch_requests"])
    return [to_device(collate_v2(part, config["model"]["max_history"]), device)
            for part in partition(examples, request_limit, config["candidate_budget"])]


def task_backward(model, batches, task, config, step):
    total = sum(len(batch["task"]) for batch in batches)
    warmup = step < config["probe_warmup_steps"]
    stats = {"loss": 0., "pairwise": 0., "eligible": 0, "helpful": 0, "harmful": 0}
    if task == "search" and not warmup:
        contexts, targets, masks = [], [], []
        model.eval()
        with torch.no_grad():
            for batch in batches:
                output = model(batch, config["candidate_chunk"])
                _, target, mask = utility_targets(output, batch, config["utility_margin"])
                contexts.append(output["vg"].detach())
                targets.append(target)
                masks.append(mask)
        model.train()
        context, target, mask = torch.cat(contexts), torch.cat(targets), torch.cat(masks)
        auxiliary_logit = model.gate(context).squeeze(-1)
        auxiliary = (F.binary_cross_entropy_with_logits(auxiliary_logit[mask], target[mask])
                     if mask.any() else auxiliary_logit.sum() * 0)
        (config["utility_weight"] * auxiliary).backward()
        stats.update(eligible=int(mask.sum()), helpful=int((target[mask] == 1).sum()),
                     harmful=int((target[mask] == 0).sum()), auxiliary=float(auxiliary.detach()))
    model.train()
    for batch in batches:
        output = model(batch, config["candidate_chunk"], gate_override=.5 if warmup else None)
        base = group_bce(output["base"], batch).mean()
        always = group_bce(output["always"], batch).mean()
        if warmup:
            loss = .5 * (base + always)
            pair = (.5 * (pairwise_loss(output["base"], batch, config["pairwise_cap"])
                          + pairwise_loss(output["always"], batch, config["pairwise_cap"]))
                    if task == "search" else loss * 0)
        else:
            loss = group_bce(output["final"], batch).mean() + config["probe_weight"] * (base + always)
            pair = pairwise_loss(output["final"], batch, config["pairwise_cap"]) if task == "search" else loss * 0
        objective = loss + config["pairwise_weight"] * pair
        if not torch.isfinite(objective):
            raise FloatingPointError(f"Non-finite {task} loss at step {step}")
        weight = len(batch["task"]) / total
        (.5 * weight * objective).backward()
        stats["loss"] += weight * float(loss.detach())
        stats["pairwise"] += weight * float(pair.detach())
    return stats


def score_rows(model, dataset, config, device):
    model.eval()
    with torch.inference_mode():
        for task in ("search", "recommendation"):
            buffer = []
            def emit(examples):
                for batch in _batches(examples, config, device):
                    output = model(batch, config["candidate_chunk"])
                    scores = output["final"].cpu().tolist()
                    offset = 0
                    for row, length in zip(batch["raw"], batch["lengths"]):
                        values = scores[offset:offset + length]
                        offset += length
                        common = {"task": task, "user_raw": row["user_raw"], "request_id": row["request_id"],
                                  "cohort": row.get("cohort", "ambiguous"), "fallback": False}
                        if task == "search":
                            yield {**common, "item_raw": row["item_raw"], "labels": row["labels"], "scores": values}
                        else:
                            for item, label, score in zip(row["item_raw"], row["labels"], values):
                                yield {**common, "item_raw": item, "label": label, "score": score}
            for row in dataset.iter_requests("validation", task):
                buffer.append(row)
                if len(buffer) == 32:
                    yield from emit(buffer)
                    buffer = []
            if buffer:
                yield from emit(buffer)


def validate(model, dataset, config, device, output_path):
    from src.evaluation.evaluate import evaluate_search, evaluate_recommendation
    search, labels, scores = [], [], []
    temporary = output_path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in score_rows(model, dataset, config, device):
            handle.write(json.dumps(row, allow_nan=False) + "\n")
            if row["task"] == "search":
                search.append(row)
            else:
                labels.append(row["label"])
                scores.append(row["score"])
    temporary.replace(output_path)
    search_bce = float(np.mean([np.mean(np.logaddexp(0, r["scores"]) - np.asarray(r["labels"]) * r["scores"])
                                for r in search]))
    rec_bce = float(np.mean(np.logaddexp(0, scores) - np.asarray(labels) * scores))
    return {"joint_log_loss": .5 * (search_bce + rec_bce), "search_group_log_loss": search_bce,
            "recommendation_log_loss": rec_bce, "search": evaluate_search(search),
            "recommendation": evaluate_recommendation(labels, scores)}


def export_weights(model, dataset, config, path):
    """Inference-only tensor export. No RNG, optimizer or training ledger."""
    temporary = path.with_suffix(".tmp")
    torch.save({"format": "trace-v2-inference-v1", "architecture": config["model"],
                "item_count": dataset.item_count, "token_count": dataset.token_count,
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "data_hash": dataset.data_hash,
                "feature_version": "trace-input-v2.0-lexical-frequency-zero-token-oov"}, temporary)
    temporary.replace(path)


def load_export(path, device="cpu"):
    value = torch.load(path, map_location="cpu", weights_only=True)
    if value["format"] != "trace-v2-inference-v1":
        raise ValueError("Unsupported weights export")
    model = TraceV2(value["item_count"], value["token_count"], **value["architecture"])
    model.load_state_dict(value["state_dict"])
    return model.to(device).eval()


def train(dataset, config, output_root, resume=None, stop_after=None, source_hash="development", job_start=None):
    """One declared experiment; stop_after is a fixture/resume tool, not tuning."""
    job_start = time.perf_counter() if job_start is None else job_start
    torch.set_num_threads(config["threads"])
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    cuda = config.get("device", "cuda") == "cuda"
    if cuda and not torch.cuda.is_available():
        raise RuntimeError("Cloud training requires CUDA; no silent CPU fallback")
    devices = [torch.device("cuda:0" if cuda else "cpu")]
    if cuda:
        torch.cuda.manual_seed_all(config["seed"])
        if torch.cuda.device_count() >= 2 and config["gpu_policy"] == "paired_tasks_two_devices_shared_model_gradient_sum":
            devices.append(torch.device("cuda:1"))
    torch.use_deterministic_algorithms(not cuda)
    model = TraceV2(dataset.item_count, dataset.token_count, **config["model"]).to(devices[0])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    warmup = max(1, math.ceil(config["steps"] * .05))
    def lr_factor(step):
        if step < warmup:
            return (step + 1) / warmup
        return .5 * (1 + math.cos(math.pi * min(1., (step - warmup) / max(1, config["steps"] - warmup))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    sampler = V2Sampler(dataset, config["seed"] + 1)
    run = Path(output_root) / config["run_id"]
    run.mkdir(parents=True, exist_ok=True)
    experiment = {k: v for k, v in config.items() if k not in ("job_soft_seconds", "finalization_reserve_seconds")}
    config_hash = object_hash({"experiment": experiment, "source_hash": source_hash, "devices": len(devices)})
    step, best, history = 0, float("inf"), []
    seen = {"search": 0, "recommendation": 0, "search_candidates": 0}
    if resume:
        snapshot = load_checkpoint(resume, models=model, optimizer=optimizer, scheduler=scheduler,
            config_hash=config_hash, data_hash=dataset.data_hash, expected_run_id=config["run_id"], map_location=devices[0])
        sampler.load_state_dict(snapshot["sampler"])
        step = snapshot["step"]
        extra = snapshot["extra"]
        best = float("inf") if extra["best"] is None else extra["best"]
        history, seen = extra["history"], extra["seen"]
    elif (run / "latest.pt").exists():
        raise FileExistsError("Existing run needs explicit resume")
    replica = copy.deepcopy(model).to(devices[1]) if len(devices) == 2 else None
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    write_json(run / "config.json", {"config": config, "config_hash": config_hash, "data_hash": dataset.data_hash,
        "source_hash": source_hash, "devices": [str(d) for d in devices], "torch": str(torch.__version__),
        "gpu_names": [torch.cuda.get_device_name(d) for d in devices] if cuda else [],
        "selection": config["selection"], "test_accessed": False})
    start = last_save = time.perf_counter()
    warm_durations, joint_durations = [], []
    profile_started = None
    profile_written = False
    best_step = None
    status = "complete"
    target = min(config["steps"], stop_after) if stop_after is not None else config["steps"]
    if cuda:
        for device in devices:
            torch.cuda.reset_peak_memory_stats(device)

    def memory():
        return {str(d): {"peak_allocated": torch.cuda.max_memory_allocated(d),
                         "peak_reserved": torch.cuda.max_memory_reserved(d),
                         "total": torch.cuda.get_device_properties(d).total_memory} for d in devices} if cuda else {}

    def save(path):
        save_checkpoint(path, models=model, optimizer=optimizer, scheduler=scheduler, step=step,
            sampler_state=sampler.state_dict(), config_hash=config_hash, data_hash=dataset.data_hash,
            config={**config, "device": "cuda" if cuda else "cpu"},
            extra={"best": None if math.isinf(best) else best, "history": history, "seen": seen})

    def validation():
        nonlocal best, best_step
        tick = time.perf_counter()
        metrics = validate(model, dataset, config, devices[0], run / "validation-latest.jsonl")
        if not math.isfinite(metrics["joint_log_loss"]):
            raise FloatingPointError("Non-finite validation selection loss")
        history.append({"step": step, "validation": metrics, "validation_seconds": time.perf_counter() - tick})
        if metrics["joint_log_loss"] < best:
            best, best_step = metrics["joint_log_loss"], step
            save(run / "best.pt")
            export_weights(model, dataset, config, run / "model-best.pt")
            write_json(run / "validation-best.json", {"step": step, **metrics})
            import shutil
            shutil.copyfile(run / "validation-latest.jsonl", run / "validation-best.jsonl")
        write_json(run / "validation-history.json", history)
        print(json.dumps({"step": step, "validation_joint_bce": metrics["joint_log_loss"]}), flush=True)

    def profile(phase, durations):
        if not durations:
            return
        mean = float(np.mean(durations))
        write_json(run / f"profile-{phase}.json", {"phase": phase, "updates": len(durations),
            "mean_update_seconds": mean, "p95_update_seconds": float(np.percentile(durations, 95)),
            "search_groups_per_second": config["search_batch"] / mean,
            "recommendation_exposures_per_second": config["recommendation_batch"] / mean,
            "remaining_optimizer_seconds_estimate": max(0, config["steps"] - step) * mean,
            "estimate_excludes_validation_checkpoint_setup": True, "cuda_memory": memory(),
            "rss_bytes_observed": peak_rss, "gpu_policy": config["gpu_policy"],
            "devices_used": len(devices), "profiling_updates_count_toward_training": True})

    save(run / "latest.pt")
    pool = ThreadPoolExecutor(max_workers=2) if replica is not None else None
    try:
        while step < target:
            if time.perf_counter() - job_start >= config["job_soft_seconds"] - config["finalization_reserve_seconds"]:
                status = "time_budget_reached"
                break
            tick = time.perf_counter()
            examples = {"search": sampler.batch("search", config["search_batch"]),
                        "recommendation": sampler.batch("recommendation", config["recommendation_batch"])}
            batches = {task: _batches(rows, config, devices[0 if task == "search" else len(devices) - 1])
                       for task, rows in examples.items()}
            model.zero_grad(set_to_none=True)
            if replica is not None:
                replica.zero_grad(set_to_none=True)
                rec_future = pool.submit(task_backward, replica, batches["recommendation"], "recommendation", config, step)
                search_stats = task_backward(model, batches["search"], "search", config, step)
                rec_stats = rec_future.result()
                for primary, secondary in zip(model.parameters(), replica.parameters()):
                    if secondary.grad is not None:
                        transferred = secondary.grad.to(devices[0])
                        if primary.grad is None:
                            primary.grad = transferred
                        else:
                            primary.grad.add_(transferred)
            else:
                search_stats = task_backward(model, batches["search"], "search", config, step)
                rec_stats = task_backward(model, batches["recommendation"], "recommendation", config, step)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
            if not torch.isfinite(grad_norm):
                raise FloatingPointError("Non-finite gradients")
            optimizer.step()
            scheduler.step()
            if replica is not None:
                replica.load_state_dict(model.state_dict())
            if cuda:
                for device in devices:
                    torch.cuda.synchronize(device)
            duration = time.perf_counter() - tick
            step += 1
            seen["search"] += config["search_batch"]
            seen["recommendation"] += config["recommendation_batch"]
            seen["search_candidates"] += sum(len(r["item_ids"]) for r in examples["search"])
            peak_rss = max(peak_rss, process.memory_info().rss)
            if step > config["profile_warmup_updates"] and step <= config["probe_warmup_steps"]:
                if profile_started is None:
                    profile_started = time.perf_counter()
                if not profile_written:
                    warm_durations.append(duration)
                    if len(warm_durations) >= config["profile_updates"] or time.perf_counter() - profile_started >= config["profile_seconds"]:
                        profile("warmup", warm_durations)
                        profile_written = True
                        print(json.dumps({"profile": "warmup_done", "step": step, "training_continues": True}), flush=True)
            if config["probe_warmup_steps"] + config["profile_warmup_updates"] < step <= config["probe_warmup_steps"] + config["profile_warmup_updates"] + config["profile_updates"]:
                joint_durations.append(duration)
                if len(joint_durations) == config["profile_updates"]:
                    profile("joint", joint_durations)
            if step == 1 or step % 50 == 0 or step == target:
                record = {"step": step, "update_seconds": duration, "lr": optimizer.param_groups[0]["lr"],
                          "search": search_stats, "recommendation": rec_stats, "presentations": seen.copy()}
                history.append(record)
                print(json.dumps(record, allow_nan=False), flush=True)
            if step % config["checkpoint_every"] == 0 or time.perf_counter() - last_save >= config["checkpoint_seconds"]:
                save(run / "latest.pt")
                last_save = time.perf_counter()
            if step % config["validation_every"] == 0:
                validation()
                save(run / "latest.pt")
        if step < config["steps"] and status == "complete":
            status = "fixture_pause" if stop_after is not None else "incomplete"
        # Reserve covers full validation/export; bounded job exits normally.
        if not any(record.get("step") == step and "validation" in record for record in history):
            validation()
        save(run / "latest.pt")
        export_weights(model, dataset, config, run / "model-latest.pt")
        if not profile_written:
            profile("warmup", warm_durations)
        if joint_durations:
            profile("joint", joint_durations)
        write_json(run / "inference-features.json", dataset.feature_state())
        report = {"status": status, "completed_steps": step, "planned_steps": config["steps"],
            "presentations": seen, "best_joint_bce": best, "best_step_this_attempt": best_step,
            "train_attempt_seconds": time.perf_counter() - start, "job_elapsed_seconds": time.perf_counter() - job_start,
            "cuda_memory": memory(), "peak_rss_bytes_observed": peak_rss, "test_accessed": False,
            "config_hash": config_hash, "data_hash": dataset.data_hash,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "parameter_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
            "embedding_bytes": sum(e.weight.numel() * e.weight.element_size() for e in (model.item_embedding, model.token_embedding))}
        write_json(run / "metrics.json", report)
        return report
    except BaseException as error:
        # Keep preceding atomic snapshot, not a partially updated optimizer/RNG.
        write_json(run / "failure.json", {"step_last_completed_in_memory": step, "error_type": type(error).__name__,
            "error": str(error), "resume_from": "latest.pt", "cuda_memory": memory()})
        raise
    finally:
        if pool is not None:
            pool.shutdown(wait=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-hash", required=True)
    parser.add_argument("--resume")
    args = parser.parse_args()
    start = time.perf_counter()
    config = read_json(Path(args.config))
    dataset = V2Dataset(args.data, config["block_cache_size"])
    if dataset.data_hash != config["prepared_data_hash"]:
        raise ValueError("Frozen prepared dataset hash mismatch")
    setup = {"dataset_load_and_verify_seconds": time.perf_counter() - start,
             "train_search_groups": len(dataset.indexes[("train", "search")]), "train_rec_exposures": dataset.rec_count,
             "arrow_bytes": dataset.rows.nbytes + dataset.blocks.nbytes + dataset.metadata.nbytes}
    write_json(Path(args.output) / config["run_id"] / "data-profile.json", setup)
    train(dataset, config, args.output, resume=args.resume, source_hash=args.source_hash, job_start=start)


if __name__ == "__main__":
    main()
