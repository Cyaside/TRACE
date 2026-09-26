"""Train-only, safe-feature TRACE baselines for logged exposures.

Popularity counts exposures, never click-through rates. Current query/caption
matching uses training-built token IDs. Histories contain permitted identities
only. Logistic/tree weights balance search request means and rec exposure means.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time
from typing import Mapping, Sequence

import numpy as np
import psutil
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

FEATURE_POLICY = "trace-safe-tabular-v1-identities-only"


def _is_search(record: Mapping) -> bool:
    task = record.get("task", "search")
    if task not in ("search", "rec", "recommendation"):
        raise ValueError(f"unsupported task {task!r}")
    return task == "search"


def _raw_key(value: object) -> str:
    return str(value)


def _tokens(values) -> set[int]:
    """Accept flat or pooled nested query lists; discard PAD=0 and OOV=1."""
    result = set()
    for value in values or []:
        if isinstance(value, (list, tuple)):
            result.update(_tokens(value))
        elif int(value) > 1:
            result.add(int(value))
    return result


def _jaccard(left: set, right: set) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _validate_records(records: Sequence[Mapping], training: bool = False) -> None:
    for index, record in enumerate(records):
        _is_search(record)
        if training and record.get("split", "train") != "train":
            raise ValueError("fit accepts training records only")
        lengths = [len(record[key]) for key in ("item_raw", "item_ids", "labels", "caption_ids")]
        if not lengths[0] or len(set(lengths)) != 1:
            raise ValueError(f"record {index}: candidate arrays must match and be nonempty")
        if _is_search(record) and len(set(map(_raw_key, record["item_raw"]))) != lengths[0]:
            raise ValueError(f"record {index}: duplicate candidates must be removed upstream")
        if any(label not in (0, 1) for label in record["labels"]):
            raise ValueError(f"record {index}: binary labels required")
        if "metadata_present" in record and len(record["metadata_present"]) != lengths[0]:
            raise ValueError(f"record {index}: metadata_present length mismatch")


class SafeFeatureExtractor:
    """Fitted exposure frequencies plus fixed inference-available features."""

    def __init__(self, id_only: bool = False):
        self.id_only = id_only
        self.feature_names = ["log_exposure_frequency", "query_token_count",
                              "query_prior_query_jaccard", "candidate_in_prior_rec",
                              "candidate_prior_rec_fraction", "history_block_count",
                              "request_gap_bucket", "prior_search_blocks_fraction",
                              "prior_rec_blocks_fraction", "is_search"]
        if not id_only:
            self.feature_names += ["query_caption_jaccard", "query_caption_intersection",
                                   "caption_token_count", "metadata_present"]
        self.frequency = Counter()
        self.fitted = False

    def fit(self, train_records: Sequence[Mapping]) -> "SafeFeatureExtractor":
        _validate_records(train_records, training=True)
        self.frequency = Counter(_raw_key(item) for record in train_records for item in record["item_raw"])
        self.fitted = True
        return self

    def request_features(self, record: Mapping) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("fit training exposure frequencies before prediction")
        query = _tokens(record.get("query_ids", []))
        history = record.get("history", [])
        prior_queries = set()
        prior_rec = Counter()
        search_blocks = rec_blocks = 0
        for block in history:
            prior_queries.update(_tokens(block.get("search_query_ids", [])))
            rec_items = _tokens(block.get("rec_item_ids", []))
            prior_rec.update(rec_items)
            mask = int(block.get("service_mask", 0))
            search_blocks += bool(mask & 1)
            rec_blocks += bool(mask & 2)
        # request_gap is a precomputed bucket, never a feedback delay feature.
        gap = float(record.get("request_gap", 0))
        if not math.isfinite(gap) or gap < 0:
            raise ValueError("request_gap must be a finite nonnegative bucket")
        rows = []
        for index, raw_item in enumerate(record["item_raw"]):
            item_id = int(record["item_ids"][index])
            appeared = prior_rec.get(item_id, 0) if item_id > 1 else 0
            features = [math.log1p(self.frequency[_raw_key(raw_item)]), len(query),
                        _jaccard(query, prior_queries), float(appeared > 0),
                        appeared / max(1, len(history)), len(history), gap,
                        search_blocks / max(1, len(history)),
                        rec_blocks / max(1, len(history)), float(_is_search(record))]
            if not self.id_only:
                caption = _tokens(record["caption_ids"][index])
                present = record.get("metadata_present", [bool(values) for values in record["caption_ids"]])[index]
                features += [_jaccard(query, caption), len(query & caption), len(caption), float(present)]
            rows.append(features)
        return np.asarray(rows, dtype=np.float64)

    def transform(self, records: Sequence[Mapping]) -> np.ndarray:
        _validate_records(records)
        if not records:
            return np.empty((0, len(self.feature_names)), dtype=np.float64)
        return np.concatenate([self.request_features(record) for record in records], axis=0)


def task_balanced_weights(records: Sequence[Mapping]) -> np.ndarray:
    """Mean row weight 1; task loss is .5 mean-search-group + .5 mean-rec-row."""
    search_groups = sum(_is_search(record) for record in records)
    rec_rows = sum(len(record["labels"]) for record in records if not _is_search(record))
    count = sum(len(record["labels"]) for record in records)
    if not count:
        raise ValueError("training records must not be empty")
    tasks = int(search_groups > 0) + int(rec_rows > 0)
    weights = []
    for record in records:
        size = len(record["labels"])
        weight = (1 / (tasks * search_groups * size) if _is_search(record)
                  else 1 / (tasks * rec_rows))
        weights.extend([weight * count] * size)
    return np.asarray(weights, dtype=np.float64)


class CheapBaseline:
    """Equal-unit sum of normalized log exposure popularity and token Jaccard."""

    def __init__(self, id_only: bool = False, seed: int = 42):
        self.extractor = SafeFeatureExtractor(id_only)
        self.id_only = id_only
        self.seed = seed
        self.maximum_log_frequency = 1.0
        self.task_intercept = {"search": 0.0, "rec": 0.0}

    def fit(self, train_records: Sequence[Mapping]) -> "CheapBaseline":
        self.extractor.fit(train_records)
        self.maximum_log_frequency = max(1.0, math.log1p(max(self.extractor.frequency.values(), default=0)))
        search_means = [sum(row["labels"]) / len(row["labels"]) for row in train_records if _is_search(row)]
        rec_labels = [label for row in train_records if not _is_search(row) for label in row["labels"]]
        for task, values in (("search", search_means), ("rec", rec_labels)):
            # A single training-only task prior is not an item CTR feature.
            prevalence = min(1 - 1e-6, max(1e-6, sum(values) / len(values))) if values else 0.5
            self.task_intercept[task] = math.log(prevalence / (1 - prevalence))
        return self

    def component_scores(self, record: Mapping) -> dict[str, list[float]]:
        features = self.extractor.request_features(record)
        popularity = features[:, 0] / self.maximum_log_frequency
        overlap = np.zeros(len(features)) if self.id_only else features[:, self.extractor.feature_names.index("query_caption_jaccard")]
        ranking_scores = popularity + overlap
        intercept = self.task_intercept["search" if _is_search(record) else "rec"]
        return {"popularity_scores": popularity.tolist(), "token_overlap_scores": overlap.tolist(),
                "ranking_scores": ranking_scores.tolist(), "scores": (ranking_scores + intercept).tolist()}

    def predict_request(self, record: Mapping) -> list[float]:
        return self.component_scores(record)["scores"]


class TabularBaseline:
    """Fixed-budget shared logistic or histogram-gradient-boosted classifier."""

    def __init__(self, model: str = "logistic", id_only: bool = False, seed: int = 42):
        if model not in ("logistic", "trees"):
            raise ValueError("model must be logistic or trees")
        self.model_name, self.id_only, self.seed = model, id_only, seed
        self.extractor = SafeFeatureExtractor(id_only)
        self.scaler = StandardScaler() if model == "logistic" else None
        self.classifier = (LogisticRegression(C=1.0, max_iter=300, random_state=seed)
                           if model == "logistic" else
                           HistGradientBoostingClassifier(max_iter=60, max_leaf_nodes=15,
                                                          learning_rate=0.1, early_stopping=False,
                                                          random_state=seed))

    def fit(self, train_records: Sequence[Mapping]) -> "TabularBaseline":
        features = self.extractor.fit(train_records).transform(train_records)
        labels = np.asarray([label for record in train_records for label in record["labels"]], dtype=np.int64)
        if set(labels.tolist()) != {0, 1}:
            raise ValueError("training population must contain both labels")
        weights = task_balanced_weights(train_records)
        if self.scaler is not None:
            self.scaler.fit(features, sample_weight=weights)
            features = self.scaler.transform(features)
        self.classifier.fit(features, labels, sample_weight=weights)
        return self

    def predict_request(self, record: Mapping) -> list[float]:
        features = self.extractor.request_features(record)
        if self.scaler is not None:
            features = self.scaler.transform(features)
        scores = np.asarray(self.classifier.decision_function(features), dtype=np.float64)
        if not np.isfinite(scores).all():
            raise RuntimeError("classifier produced nonfinite logits")
        return scores.tolist()


def prediction_records(model, records: Sequence[Mapping]) -> list[dict]:
    """Save original validation candidates/labels without test access or fitting."""
    _validate_records(records)
    result = []
    for record in records:
        scores = model.predict_request(record)
        common = {"user_raw": str(record["user_raw"]), "request_id": str(record["request_id"]),
                  "cohort": record["cohort"], "timestamp_ms": record["timestamp_ms"],
                  "fallback": False}
        if _is_search(record):
            row = {**common, "task": "search", "item_raw": list(map(str, record["item_raw"])),
                   "labels": list(map(int, record["labels"])), "scores": scores,
                   "metadata_present": record.get("metadata_present", [None] * len(scores))}
            if isinstance(model, CheapBaseline):
                row.update(model.component_scores(record))
            result.append(row)
        else:
            for index, score in enumerate(scores):
                result.append({**common, "task": "recommendation",
                               "request_id": f"{record['request_id']}:{index}",
                               "item_raw": str(record["item_raw"][index]),
                               "label": int(record["labels"][index]), "score": score})
    return result


def _hash_records(records: Sequence[Mapping]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(json.dumps(record, sort_keys=True, separators=(",", ":"), default=int).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _run(args, dataset, output: Path, started: float) -> dict:
    from src.evaluation.evaluate import evaluate_predictions
    from src.training.ledger import provenance
    from src.training.serialization import serialize_tabular_model
    train = dataset.requests("train")
    validation = dataset.requests("validation")
    # No test requests are encoded, scored, or selected during this command.
    model = (CheapBaseline(args.id_only, args.seed) if args.model == "cheap"
             else TabularBaseline(args.model, args.id_only, args.seed))
    fit_start = time.perf_counter()
    model.fit(train)
    fit_seconds = time.perf_counter() - fit_start
    predictions = prediction_records(model, validation)
    metrics = evaluate_predictions(predictions)
    feature_policy = {"version": FEATURE_POLICY, "id_only": args.id_only,
                      "features": model.extractor.feature_names,
                      "popularity": "training logged exposure counts; no outcomes",
                      "history": "strictly prior identity blocks; no historical feedback",
                      "task_weighting": "equal search-group and recommendation-exposure task means",
                      "scaling": "training task-weighted StandardScaler" if args.model == "logistic" else None}
    source_hashes = {str(path.relative_to(args.data)): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in sorted(args.data.glob("*.json"))}
    manifest = {**provenance(dataset.manifest),
                "run_id": args.run_id, "model": args.model, "seed": args.seed,"data_hash":dataset.data_hash,
                "parent_run":None,"question":"safe-feature and metadata-sensitivity development baseline",
                "data_release_hash":dataset.manifest.get('raw_archive_sha256'),"teacher_hash":None,
                "optimizer_updates":None,"precision":"float64 sklearn features/logits",
                "created_at": datetime.now(timezone.utc).isoformat(), "status": "complete",
                "split": "validation", "data_root": str(args.data.resolve()),
                "source_manifest_hashes": source_hashes,
                "training_records_sha256": _hash_records(train),
                "validation_records_sha256": _hash_records(validation),
                "feature_policy": feature_policy,
                "feature_policy_hash": hashlib.sha256(json.dumps(feature_policy, sort_keys=True).encode()).hexdigest(),
                "prepared_feature_policy_hash": dataset.manifest.get("feature_policy_hash"),
                "training_requests": len(train),
                "training_candidates": sum(len(record["labels"]) for record in train),
                "validation_requests": len(validation), "row_cap": None, "candidate_cap": None,
                "fit_seconds": fit_seconds, "wall_time_seconds": time.perf_counter() - started,
                "processed_examples":sum(len(record['labels']) for record in train),
                "interpretation":"bounded single-seed validation baseline; no convergence or research-gain claim",
                "hardware": "CPU", "candidate_policy": "all supplied logged candidates; no truncation",
                "peak_memory_bytes": getattr(psutil.Process().memory_info(), "peak_wset", psutil.Process().memory_info().rss),
                "software": {"python": sys.version.split()[0], "numpy": np.__version__, "scikit_learn": sklearn.__version__},
                "validation_metrics": metrics,
                "cheap_calibration": "training-only global task-prevalence intercept; fixed unit slope; not calibrated click probabilities"
                                     if args.model == "cheap" else None,
                "model_config": {"C": 1.0, "max_iter": 300} if args.model == "logistic"
                                else {"max_iter": 60, "max_leaf_nodes": 15, "early_stopping": False}
                                if args.model == "trees" else {"score": "normalized log exposure frequency + token Jaccard"}}
    output.mkdir(parents=True)
    (output / "model.pkl").write_bytes(serialize_tabular_model(model))
    (output / "validation_predictions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in predictions), encoding="utf-8")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model", choices=("cheap", "logistic", "trees"), required=True)
    parser.add_argument("--id-only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_id):
        parser.error("run-id must be a simple filename beginning with a letter or digit")
    from src.common import ARTIFACTS_ROOT
    from src.features.dataset import PreparedDataset
    from src.training.ledger import append_run, provenance
    started = time.perf_counter()
    output = ARTIFACTS_ROOT / "runs" / args.run_id
    if output.exists():
        raise FileExistsError(f"run directory already exists: {output}")
    event = {"run_id": args.run_id, "model": args.model, "seed": args.seed,
             "id_only": args.id_only, "data_root": str(args.data.resolve()),
             "question": "safe-feature logged-exposure baseline", "parent_run": None,
             "status": "started", **provenance()}
    append_run(event)
    try:
        dataset = PreparedDataset(args.data)
        manifest = _run(args, dataset, output, started)
    except Exception as exc:
        append_run({**event, "status": "failed", "error_type": type(exc).__name__,
                    "error": str(exc), "wall_time_seconds": time.perf_counter() - started})
        raise
    append_run({**manifest, "status": "complete"})
    print(json.dumps({"run_id": args.run_id, "output": str(output), "fit_seconds": manifest["fit_seconds"],
                      "validation_metrics": manifest["validation_metrics"]}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
