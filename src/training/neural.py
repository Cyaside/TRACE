"""Bounded baseline training; importing this module never starts a run."""
from __future__ import annotations

import argparse
import bisect
import dataclasses
import json
import math
import platform
import random
import time
import uuid
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import psutil
import torch
from torch import nn

from src.common import ARTIFACTS_ROOT, object_hash, read_json, write_json
from src.models.neural import CompactRanker, collate_requests, supervised_loss, to_device
from src.training.checkpoint import load_checkpoint, save_checkpoint


@dataclasses.dataclass
class TrainConfig:
    run_id: str = "shared-pilot"
    mode: str = "shared"
    steps: int = 1000
    batch_size: int = 32
    lr: float = .001
    weight_decay: float = .0001
    seed: int = 42
    device: str = "cpu"
    threads: int = 4
    max_seconds: float | None = 600
    profile_steps: int = 0
    validation_every: int = 250
    checkpoint_seconds: float = 300
    resume: str | None = None
    output_root: str = str(ARTIFACTS_ROOT / "runs")
    embedding_dim: int = 32
    hidden_dim: int = 64
    heads: int = 2
    layers: int = 1
    ff_dim: int = 128
    dropout: float = .1
    max_history: int = 50
    candidate_chunk_size: int | None = None
    max_padded_candidates: int = 8192
    amp: bool = False
    record_ledger: bool = True
    parent_run: str | None = None
    question: str = "How do supervised separate-task and shared block encoders compare under the same inputs and update budget?"

    def experiment_dict(self) -> dict[str, Any]:
        # Operational limits do not change the optimizer schedule or sampled stream.
        excluded = {"run_id", "max_seconds", "profile_steps", "resume", "output_root", "checkpoint_seconds",
                    "record_ledger", "parent_run", "question"}
        return {key: value for key, value in dataclasses.asdict(self).items() if key not in excluded}


class TaskSampler:
    """Uniform search groups and uniform recommendation exposures, with replacement."""

    def __init__(self, search: list[dict], recommendation: list[dict], seed: int):
        if not search or not recommendation:
            raise ValueError("Joint training needs both search groups and recommendation exposures.")
        self.search, self.recommendation = search, recommendation
        self.rng = random.Random(seed)
        self.cursor = 0
        self.rec_offsets = []
        total = 0
        for request in recommendation:
            if not request["item_ids"]:
                raise ValueError("Recommendation requests cannot contain empty exposure lists.")
            total += len(request["item_ids"])
            self.rec_offsets.append(total)
        self.rec_count = total

    def batch(self, task: str, size: int) -> list[dict]:
        if task == "search":
            result = [self.search[self.rng.randrange(len(self.search))] for _ in range(size)]
        elif task == "recommendation":
            result = []
            for _ in range(size):
                exposure = self.rng.randrange(self.rec_count)
                row = bisect.bisect_right(self.rec_offsets, exposure)
                column = exposure - (self.rec_offsets[row - 1] if row else 0)
                request = self.recommendation[row]
                example = dict(request)
                for key in ("item_ids", "caption_ids", "labels", "item_raw"):
                    if key in request:
                        example[key] = [request[key][column]]
                result.append(example)
        else:
            raise ValueError("Unknown sampler task.")
        self.cursor += size
        return result

    def state_dict(self):
        return {"rng": self.rng.getstate(), "cursor": self.cursor}

    def load_state_dict(self, state):
        self.rng.setstate(state["rng"])
        self.cursor = int(state["cursor"])


def build_models(item_count: int, token_count: int, config: TrainConfig) -> nn.ModuleDict:
    architecture = {key: getattr(config, key) for key in (
        "embedding_dim", "hidden_dim", "heads", "layers", "ff_dim", "dropout", "max_history")}
    if config.mode == "shared":
        return nn.ModuleDict({"shared": CompactRanker(item_count, token_count, **architecture)})
    if config.mode == "separate":
        return nn.ModuleDict({task: CompactRanker(item_count, token_count, history_mode="task_only", **architecture)
                              for task in ("search", "recommendation")})
    raise ValueError("Mode must be shared or separate.")


def _model(models: nn.ModuleDict, task: str):
    return models["shared"] if "shared" in models else models[task]


def microbatches(examples, max_padded_candidates: int = 8192):
    """Keep complete slates while bounding waste from padding unequal groups.

    A single group larger than the budget is retained whole; the budget is a
    batching limit, not candidate truncation. Group-count weighting in training
    preserves the original search group mean after this partition.
    """
    if max_padded_candidates < 1:
        raise ValueError("Padded candidate budget must be positive.")
    current, longest = [], 0
    for example in sorted(examples, key=lambda x: len(x["item_ids"])):
        size = len(example["item_ids"])
        next_longest = max(longest, size)
        if current and (len(current) + 1) * next_longest > max_padded_candidates:
            yield current
            current, longest = [], 0
        current.append(example)
        longest = max(longest, size)
    if current:
        yield current


def score_requests(models: nn.ModuleDict, requests, device="cpu", batch_size: int = 32,
                   candidate_chunk_size: int | None = None,
                   max_padded_candidates: int = 8192) -> Iterator[dict]:
    """Score unchanged logged candidates; emit the evaluator's canonical row schema."""
    models.eval()
    with torch.inference_mode():
        for start in range(0, len(requests), batch_size):
            batch_requests = requests[start:start + batch_size]
            # Separate models cannot process the other task with their own histories.
            for task in ("search", "recommendation"):
                selected = [x for x in batch_requests if x["task"] == task]
                if not selected:
                    continue
                model = _model(models, task)
                for part in microbatches(selected, max_padded_candidates):
                    batch = to_device(collate_requests(part, model.max_history), device)
                    scores = model(batch, candidate_chunk_size).cpu().tolist()
                    for request, values in zip(part, scores):
                        count = len(request["item_ids"])
                        common = {"task": task, "user_raw": request["user_raw"],
                                  "request_id": request["request_id"],
                                  "cohort": request.get("cohort", "ambiguous")}
                        if task == "search":
                            yield {**common, "item_raw": request["item_raw"],
                                   "labels": request["labels"], "scores": values[:count],
                                   "fallback": False}
                        else:
                            for item, label, value in zip(request["item_raw"], request["labels"], values[:count]):
                                yield {**common, "item_raw": item, "label": label, "score": value,
                                       "fallback": False}


def validation_metrics(models, requests, device, batch_size, candidate_chunk_size=None,
                       max_padded_candidates=8192):
    rows = list(score_requests(models, requests, device, batch_size, candidate_chunk_size, max_padded_candidates))
    search = [row for row in rows if row["task"] == "search"]
    rec = [row for row in rows if row["task"] == "recommendation"]
    def log_loss(labels, logits):
        labels, logits = np.asarray(labels, dtype=np.float64), np.asarray(logits, dtype=np.float64)
        return float(np.mean(np.logaddexp(0, logits) - labels * logits)) if len(labels) else None
    group_losses = [log_loss(x["labels"], x["scores"]) for x in search]
    search_loss = float(np.mean(group_losses)) if group_losses else None
    rec_loss = log_loss([x["label"] for x in rec], [x["score"] for x in rec])
    if search_loss is None or rec_loss is None:
        raise ValueError("Validation must contain both tasks before model selection.")
    metrics = {"search_group_log_loss": search_loss, "recommendation_log_loss": rec_loss,
               "joint_log_loss": .5 * (search_loss + rec_loss),
               "search_requests": len(search), "recommendation_exposures": len(rec)}
    try:
        from src.evaluation.evaluate import evaluate_search, evaluate_recommendation
    except ImportError:
        # Development fixtures can run before the standalone evaluator is present.
        metrics["evaluator_status"] = "not_available"
    else:
        metrics["search"] = evaluate_search(search)
        metrics["recommendation"] = evaluate_recommendation([x["label"] for x in rec], [x["score"] for x in rec])
        metrics["evaluator_status"] = "available"
    return metrics, rows


def _save_predictions(path: Path, rows):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def _train(dataset, config: TrainConfig, run_evidence: dict | None = None) -> dict:
    """Train fixed-budget baselines, stopping safely at an operational time cap.

    Best checkpoint selection uses the *validation joint BCE* with equal task
    weight. All candidates enter search BCE; recommendation sampling/reduction
    is exposure-uniform. No test split is accessed.
    """
    call_start = time.perf_counter()
    if config.steps < 1 or config.batch_size < 1 or config.threads < 1 or config.max_padded_candidates < 1:
        raise ValueError("Steps, batch size, and CPU threads must be positive.")
    if config.profile_steps < 0 or config.validation_every < 0 or config.checkpoint_seconds <= 0:
        raise ValueError("Invalid profile, validation, or checkpoint interval.")
    if config.max_seconds is not None and config.max_seconds <= 0:
        raise ValueError("Time cap must be positive or None.")
    if not config.run_id or Path(config.run_id).name != config.run_id:
        raise ValueError("Run ID must be one directory name.")
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use CPU or a cloud GPU runtime.")
    if config.amp and config.device != "cuda":
        raise ValueError("AMP is enabled only for the CUDA training option.")
    torch.set_num_threads(config.threads)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    torch.use_deterministic_algorithms(config.device == "cpu")
    run_dir = Path(config.output_root) / config.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    latest = run_dir / "latest.pt"
    if latest.exists() and not config.resume:
        raise FileExistsError(f"Run already has a checkpoint; use --resume: {latest}")
    config_hash = object_hash(config.experiment_dict())
    data_hash = str(dataset.data_hash)
    load_start = time.perf_counter()
    train_requests = list(dataset.requests("train"))
    val_requests = list(dataset.requests("validation"))
    search = [x for x in train_requests if x["task"] == "search"]
    recommendation = [x for x in train_requests if x["task"] == "recommendation"]
    sampler = TaskSampler(search, recommendation, config.seed + 1)
    data_loading_seconds = time.perf_counter() - load_start
    models = build_models(dataset.item_count, dataset.token_count, config).to(config.device)
    optimizer = torch.optim.AdamW(models.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    warmup = max(1, math.ceil(config.steps * .05))
    def learning_rate_factor(step):
        if step < warmup:
            return (step + 1) / warmup
        fraction = min(1., (step - warmup) / max(1, config.steps - warmup))
        return .5 * (1 + math.cos(math.pi * fraction))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_factor)
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp)
    step = 0
    best_joint_loss = float("inf")
    records = []
    cumulative_training_seconds = 0.
    seen_search = seen_rec = seen_search_candidates = 0
    if config.resume:
        path = latest if config.resume == "latest" else Path(config.resume)
        snapshot = load_checkpoint(path, models=models, optimizer=optimizer, scheduler=scheduler,
            config_hash=config_hash, data_hash=data_hash, scaler=scaler, map_location=config.device,
            expected_run_id=config.run_id)
        step = snapshot["step"]
        sampler.load_state_dict(snapshot["sampler"])
        extra = snapshot["extra"]
        best_joint_loss = extra.get("best_joint_loss")
        best_joint_loss = float("inf") if best_joint_loss is None else float(best_joint_loss)
        records = extra.get("records", [])
        cumulative_training_seconds = extra.get("training_seconds", 0.)
        seen_search, seen_rec = extra.get("seen_search", 0), extra.get("seen_rec", 0)
        seen_search_candidates = extra.get("seen_search_candidates", 0)
    else:
        write_json(run_dir / "config.json", {**dataclasses.asdict(config), "config_hash": config_hash,
            "data_hash": data_hash, "item_count": dataset.item_count, "token_count": dataset.token_count,
            "selection_rule": "minimum validation .5 search-group BCE + .5 recommendation-exposure BCE",
            "sampling": "with replacement; search group uniform, recommendation exposure uniform",
            "attribution": "Independent SASRec-derived causal block-history/logged-exposure adaptation",
            "provenance": run_evidence or {}})
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    if config.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start_time = last_checkpoint = time.perf_counter()
    initial_step = step
    call_training_seconds = 0.
    collation_seconds = 0.
    max_padded_slots_observed = 0
    target = min(config.steps, config.profile_steps) if config.profile_steps else config.steps
    status = "profile_complete" if target < config.steps else "complete"
    initial_loss, last_loss = None, None
    last_validation_step = -1

    def snapshot_extra():
        return {"best_joint_loss": None if math.isinf(best_joint_loss) else best_joint_loss,
                "records": records, "training_seconds": cumulative_training_seconds + call_training_seconds,
                "seen_search": seen_search, "seen_rec": seen_rec,
                "seen_search_candidates": seen_search_candidates}

    def save(path):
        save_checkpoint(path, models=models, optimizer=optimizer, scheduler=scheduler, step=step,
                        sampler_state=sampler.state_dict(), config_hash=config_hash, data_hash=data_hash,
                        config={**dataclasses.asdict(config), "provenance": run_evidence or {}},
                        scaler=scaler, extra=snapshot_extra())

    def validate():
        nonlocal best_joint_loss, last_validation_step, peak_rss
        metrics, rows = validation_metrics(models, val_requests, config.device,
                                           config.batch_size, config.candidate_chunk_size,
                                           config.max_padded_candidates)
        records.append({"step": step, "validation": metrics})
        last_validation_step = step
        peak_rss = max(peak_rss, process.memory_info().rss)
        if metrics["joint_log_loss"] < best_joint_loss:
            best_joint_loss = metrics["joint_log_loss"]
            save(run_dir / "best.pt")
            _save_predictions(run_dir / "validation-best.jsonl", rows)
            write_json(run_dir / "validation-best.json", {"step": step, **metrics})
        return metrics

    # A valid initial snapshot exists even if the first optimizer update is interrupted.
    if not config.resume:
        save(latest)
    safe_update_boundary = True
    try:
        while step < target:
            if config.max_seconds is not None and time.perf_counter() - start_time >= config.max_seconds:
                status = "time_budget_reached"
                break
            tick = time.perf_counter()
            safe_update_boundary = False
            models.train()
            optimizer.zero_grad(set_to_none=True)
            task_losses = []
            candidate_presentations = 0
            for task in ("search", "recommendation"):
                collate_tick = time.perf_counter()
                examples = sampler.batch(task, config.batch_size)
                if task == "search":
                    candidate_presentations = sum(len(x["item_ids"]) for x in examples)
                collation_seconds += time.perf_counter() - collate_tick
                task_loss = 0.
                for part in microbatches(examples, config.max_padded_candidates):
                    collate_tick = time.perf_counter()
                    batch = to_device(collate_requests(part, config.max_history), config.device)
                    collation_seconds += time.perf_counter() - collate_tick
                    max_padded_slots_observed = max(max_padded_slots_observed, batch["item_ids"].numel())
                    with torch.autocast(device_type=config.device, enabled=config.amp):
                        logits = _model(models, task)(batch, config.candidate_chunk_size)
                        loss = supervised_loss(logits.float(), batch, "group" if task == "search" else "exposure")
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Non-finite {task} loss at update {step}.")
                    fraction = len(part) / len(examples)
                    scaler.scale(.5 * fraction * loss).backward()
                    task_loss += fraction * float(loss.detach().cpu())
                task_losses.append(task_loss)
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(models.parameters(), 1.)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f"Non-finite gradient at update {step}.")
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            if config.device == "cuda":
                torch.cuda.synchronize()
            step += 1
            seen_search += config.batch_size
            seen_rec += config.batch_size
            seen_search_candidates += candidate_presentations
            duration = time.perf_counter() - tick
            call_training_seconds += duration
            safe_update_boundary = True
            last_loss = .5 * sum(task_losses)
            if initial_loss is None:
                initial_loss = last_loss
            peak_rss = max(peak_rss, process.memory_info().rss)
            if step == 1 or step % 50 == 0 or step == target:
                item = {"step": step, "train_joint_loss": last_loss,
                        "train_search_loss": task_losses[0], "train_recommendation_loss": task_losses[1],
                        "step_seconds": duration, "lr": optimizer.param_groups[0]["lr"]}
                records.append(item)
                print(json.dumps(item), flush=True)
            if config.validation_every and step % config.validation_every == 0:
                validate()
            if time.perf_counter() - last_checkpoint >= config.checkpoint_seconds:
                save(latest)
                last_checkpoint = time.perf_counter()
        if last_validation_step != step:
            validate()
    except KeyboardInterrupt:
        status = "interrupted"
        if safe_update_boundary:
            save(latest)
        raise
    except Exception:
        # A failed in-flight optimizer update is not a safe continuation point.
        # Keep the preceding atomic checkpoint rather than saving partial gradients/RNG.
        raise
    save(latest)
    elapsed = time.perf_counter() - start_time
    total_training = cumulative_training_seconds + call_training_seconds
    report = {
        "status": status, "completed_steps": step, "planned_steps": config.steps,
        "profile_target_steps": target, "config_hash": config_hash, "data_hash": data_hash,
        "parameter_count": sum(x.numel() for x in models.parameters()),
        "parameter_bytes_fp32": sum(x.numel() * x.element_size() for x in models.parameters()),
        "peak_rss_bytes_observed": peak_rss,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if config.device == "cuda" else None,
        "call_wall_seconds": time.perf_counter() - call_start,
        "training_and_validation_wall_seconds": elapsed, "training_seconds": total_training,
        "data_loading_seconds": data_loading_seconds,
        "call_collation_seconds": collation_seconds,
        "max_padded_candidate_slots_observed": max_padded_slots_observed,
        "mean_call_step_seconds": call_training_seconds / (step - initial_step) if step > initial_step else None,
        "estimated_1000_steps_seconds": call_training_seconds * 1000 / (step - initial_step) if step > initial_step else None,
        "search_group_presentations": seen_search, "recommendation_exposure_presentations": seen_rec,
        "search_candidate_presentations": seen_search_candidates,
        "candidate_presentations_per_training_second": (seen_search_candidates + seen_rec) / total_training if total_training else None,
        "presentations_per_training_second": (seen_search + seen_rec) / total_training if total_training else None,
        "initial_call_train_loss": initial_loss, "last_call_train_loss": last_loss,
        "best_validation_joint_log_loss": best_joint_loss,
        "device": config.device, "threads": config.threads,
        "cpu": platform.processor(), "platform": platform.platform(),
        "torch_version": torch.__version__, "records": records,
        "provenance": run_evidence or {},
        "time_budget_note": "Cap checked between updates; final validation/checkpoint may extend wall time.",
    }
    write_json(run_dir / "metrics.json", report)
    print(json.dumps({key: report[key] for key in ("status", "completed_steps", "call_wall_seconds", "peak_rss_bytes_observed")}), flush=True)
    return report


def train(dataset, config: TrainConfig) -> dict:
    """Run one inspectable attempt, preserving one run identity across resume."""
    from src.training.ledger import append_run, provenance
    attempt_start = time.perf_counter()
    attempt_id = uuid.uuid4().hex
    manifest = getattr(dataset, "manifest", None)
    evidence = provenance(manifest if isinstance(manifest, dict) else None)
    hardware = {
        "device": config.device, "cpu": platform.processor(), "platform": platform.platform(),
        "cpu_threads": config.threads, "system_ram_bytes": psutil.virtual_memory().total,
        "gpu": torch.cuda.get_device_name() if config.device == "cuda" and torch.cuda.is_available() else None,
        "torch_version": torch.__version__,
    }
    common = {"run_id": config.run_id, "attempt_id": attempt_id,
              "method": f"neural-{config.mode}", "config_hash": object_hash(config.experiment_dict()),
              "data_hash": str(dataset.data_hash), "seed": config.seed,
              "device": config.device, "threads": config.threads,
              "resume_from": config.resume, "provenance": evidence,
              "parent_run": config.parent_run, "question": config.question,
              **evidence,
              "data_release_hash": evidence.get("raw_archive_sha256"),
              "candidate_policy": {
                  "search": "canonical retained logged candidates; VIDEO/IMAGE_ATLAS; no sampled negatives",
                  "recommendation": "canonical logged distribution exposures",
                  "missing_metadata": "retain candidate; empty caption representation",
                  "unknown_identity": "OOV=1; retain available metadata; no dropped candidates",
                  "candidate_truncation": False,
              },
              "model_config": config.experiment_dict(),
              "optimizer_updates": None if config.resume else 0,
              "processed_examples": None if config.resume else {
                  "search_groups": 0, "search_candidates": 0, "recommendation_exposures": 0,
                  "total_group_or_exposure_presentations": 0},
              "precision": "AMP fp16 operators; fp32 parameters/loss" if config.amp else "fp32",
              "hardware": hardware, "wall_time": None, "peak_memory": None,
              "teacher_hash": None, "validation_metrics": None,
              "interpretation": "Attempt started; no terminal quality or resource conclusion yet."}
    if config.record_ledger:
        append_run({**common, "status": "started", "planned_steps": config.steps,
                    "max_seconds": config.max_seconds})
    try:
        report = _train(dataset, config, {**evidence, "attempt_id": attempt_id})
    except BaseException as error:
        if config.record_ledger:
            append_run({**common, "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                        "error_type": type(error).__name__, "error": str(error),
                        "checkpoint": str(Path(config.output_root) / config.run_id / "latest.pt"),
                        "optimizer_updates": None, "processed_examples": None,
                        "wall_time": time.perf_counter() - attempt_start,
                        "peak_memory": {"cpu_process_peak_working_set_bytes":
                            getattr(psutil.Process().memory_info(), "peak_wset", None)},
                        "interpretation": "Attempt stopped without terminal metrics; completed update counts remain in the preceding atomic checkpoint."})
        raise
    if config.record_ledger:
        best_path = Path(config.output_root) / config.run_id / "validation-best.json"
        selected_validation = read_json(best_path) if best_path.exists() else None
        append_run({**common, **{key: report[key] for key in (
            "status", "completed_steps", "call_wall_seconds", "training_seconds",
            "peak_rss_bytes_observed", "parameter_count", "best_validation_joint_log_loss")},
            "metrics_path": str(Path(config.output_root) / config.run_id / "metrics.json"),
            "optimizer_updates": report["completed_steps"],
            "processed_examples": {
                "search_groups": report["search_group_presentations"],
                "search_candidates": report["search_candidate_presentations"],
                "recommendation_exposures": report["recommendation_exposure_presentations"],
                "total_group_or_exposure_presentations": report["search_group_presentations"] + report["recommendation_exposure_presentations"],
                "sampling": "with replacement; presentations are not distinct examples",
            },
            "wall_time": report["call_wall_seconds"],
            "peak_memory": {"cpu_rss_bytes_observed": report["peak_rss_bytes_observed"],
                            "cuda_peak_allocated_bytes": report["peak_cuda_allocated_bytes"]},
            "validation_metrics": selected_validation,
            "interpretation": (
                "Completed the fixed supervised baseline budget; checkpoint selected by validation joint BCE. "
                "Matched baseline comparison is pending; no teacher, distillation, or final-test claim."
                if report["status"] == "complete" else
                "Operationally bounded partial/profile run; resume or compare only under a declared matched update budget. "
                "Validation metrics are preliminary; no final-test claim."),
        })
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mode", choices=("shared", "separate"), default="shared")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-seconds", type=float, default=600)
    parser.add_argument("--profile-steps", type=int, default=0)
    parser.add_argument("--validation-every", type=int, default=250)
    parser.add_argument("--resume", nargs="?", const="latest")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--candidate-chunk-size", type=int)
    parser.add_argument("--max-padded-candidates", type=int, default=8192)
    parser.add_argument("--parent-run")
    parser.add_argument("--question", default=TrainConfig.question)
    args = parser.parse_args()
    from src.features.dataset import PreparedDataset
    config = TrainConfig(**{key: value for key, value in vars(args).items() if key != "data"})
    train(PreparedDataset(args.data), config)


if __name__ == "__main__":
    main()
