"""Matched validation baselines and frozen-branch diagnostics for TRACE V2.

No neural training or test evaluation. Feature arrays live on disk; histories
are decoded once per request, rather than expanded into a whole Python cohort.
"""
from __future__ import annotations

import argparse
from collections import Counter, OrderedDict, defaultdict
import json
import math
from pathlib import Path
import time
import warnings

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import psutil
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from src.common import digest, object_hash, read_json, write_json
from src.evaluation.evaluate import evaluate_predictions, evaluate_search, search_request_metrics
from src.features.v2_dataset import V2Dataset
from src.training.tabular import SafeFeatureExtractor

FEATURE_VERSION = "matched-10k-safe17-raw-overlap-v1"
EXTRA_FEATURES = ["raw_query_caption_query_coverage", "raw_query_caption_caption_coverage",
                  "raw_query_caption_jaccard"]
BRANCHES = ("base", "always", "final")


def jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


class ExposureLookup:
    """String-key adapter to the reader's already fitted train-only frequencies."""
    def __init__(self, dataset):
        self.ids, self.counts = dataset.frequency_ids, dataset.frequencies

    def __getitem__(self, raw):
        value = int(raw)
        i = int(np.searchsorted(self.ids, value))
        return int(self.counts[i]) if i < len(self.ids) and self.ids[i] == value else 0


def extractor_for(dataset):
    extractor = SafeFeatureExtractor()
    extractor.frequency = ExposureLookup(dataset)
    extractor.fitted = True
    return extractor


def feature_rows(extractor, row, dataset):
    # Distinguish absent metadata from an existing row with an empty caption.
    positions = np.searchsorted(dataset.meta_ids, row["item_raw"])
    present = np.zeros(len(positions), dtype=bool)
    valid = positions < len(dataset.meta_ids)
    present[valid] = dataset.meta_ids[positions[valid]] == np.asarray(row["item_raw"])[valid]
    enriched = {**row, "metadata_present": present.tolist()}
    standard = extractor.request_features(enriched)
    # Raw overlap avoids handicapping a tabular comparator by collapsing OOV
    # tokens that the V2 lexical path itself can match. Original features remain.
    extra = np.asarray(row["relevance_features"], dtype=np.float64)[:, :3]
    return np.concatenate([standard, extra], axis=1)


class FastTabularFeatures:
    """Compact cached block sets; no expanded neural histories for baselines.

    Equivalent to feature_rows on the same request. All ID filtering still uses
    the frozen training vocab; only the additional lexical path uses raw tokens.
    """
    def __init__(self, dataset, cache_size=50000):
        self.data = dataset
        self.cache_size = cache_size
        self.blocks, self.captions = OrderedDict(), OrderedDict()

    def _cache(self, cache, key, build):
        if key in cache:
            cache.move_to_end(key)
            return cache[key]
        value = build()
        cache[key] = value
        if len(cache) > self.cache_size:
            cache.popitem(last=False)
        return value

    def block(self, index):
        def build():
            row = self.data.blocks.slice(int(index), 1).to_pylist()[0]
            # Pool queries as unordered sets, exactly as SafeFeatureExtractor.
            tokens = [t for query in row["search_queries"] for t in query[:50]]
            query_ids = frozenset(i for i in self.data.lookup(self.data.tokens, tokens) if i > 1)
            rec_ids = frozenset(i for i in self.data.lookup(self.data.items, row["rec_items"]) if i > 1)
            return query_ids, rec_ids, row["service_mask"]
        return self._cache(self.blocks, int(index), build)

    def caption(self, raw):
        def build():
            i = int(np.searchsorted(self.data.meta_ids, raw))
            present = i < len(self.data.meta_ids) and self.data.meta_ids[i] == raw
            tokens = (self.data.metadata["caption_tokens"][int(self.data.meta_order[i])].as_py()[:50]
                      if present else [])
            return (frozenset(i for i in self.data.token_ids(tokens) if i > 1),
                    frozenset(tokens), bool(present))
        return self._cache(self.captions, int(raw), build)

    def __call__(self, row):
        data = self.data
        query = set(i for i in data.token_ids(row["query_tokens"]) if i > 1)
        raw_query = set(row["query_tokens"][:50]) if row["task"] == "search" else set()
        prior_queries, prior_rec = set(), Counter()
        search_blocks = rec_blocks = 0
        history = row["history_ids"][-50:]
        for index in history:
            queries, rec, mask = self.block(index)
            prior_queries.update(queries)
            prior_rec.update(rec)
            search_blocks += bool(mask & 1)
            rec_blocks += bool(mask & 2)
        union = query | prior_queries
        history_overlap = len(query & prior_queries) / len(union) if union else 0.
        items = data.lookup(data.items, row["item_raw"])
        positions = np.searchsorted(data.frequency_ids, row["item_raw"])
        result = []
        for raw, item, position in zip(row["item_raw"], items, positions):
            frequency = int(data.frequencies[position]) if position < len(data.frequency_ids) and data.frequency_ids[position] == raw else 0
            caption, raw_caption, present = self.caption(raw)
            union = query | caption
            overlap = len(raw_query & raw_caption)
            appeared = prior_rec[item] if item > 1 else 0
            result.append([math.log1p(frequency), len(query), history_overlap, float(appeared > 0),
                appeared / max(1, len(history)), len(history), float(row["request_gap"]),
                search_blocks / max(1, len(history)), rec_blocks / max(1, len(history)), float(row["task"] == "search"),
                len(query & caption) / len(union) if union else 0., len(query & caption), len(caption), float(present),
                overlap / max(1, len(raw_query)), overlap / max(1, len(raw_caption)),
                overlap / max(1, len(raw_query | raw_caption))])
        return np.asarray(result, dtype=np.float64)


def _split_indexes(dataset, split):
    return np.sort(np.concatenate([dataset.indexes[(split, task)]
                                  for task in ("search", "recommendation")]))


def prepare_features(dataset, output):
    """Disk matrices, complete candidates, equal-task/group training weights."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    contract = {"data_hash": dataset.data_hash, "features": FEATURE_VERSION,
                "feature_names": SafeFeatureExtractor().feature_names + EXTRA_FEATURES,
                "test_accessed": False}
    manifest_path = output / "features.json"
    if manifest_path.exists():
        existing = read_json(manifest_path)
        if existing["contract"] != contract:
            raise ValueError("Existing feature cache has a different contract")
        for name, expected in existing["files"].items():
            if digest(output / name) != expected:
                raise ValueError(f"Feature cache checksum mismatch: {name}")
        return existing
    started = time.perf_counter()
    builder = FastTabularFeatures(dataset)
    info, paths = {}, []
    train_search = len(dataset.indexes[("train", "search")])
    train_rec = dataset.rec_count
    for split in ("train", "validation"):
        indexes = _split_indexes(dataset, split)
        lengths = pc.list_value_length(dataset.rows["item_raw"].take(pa.array(indexes))).to_numpy()
        count = int(lengths.sum())
        x = np.lib.format.open_memmap(output / f"{split}-x.npy", mode="w+", dtype="float64",
                                     shape=(count, len(contract["feature_names"])))
        y = np.lib.format.open_memmap(output / f"{split}-y.npy", mode="w+", dtype="int8", shape=(count,))
        w = np.lib.format.open_memmap(output / f"{split}-w.npy", mode="w+", dtype="float64", shape=(count,))
        offset = 0
        records_path = output / f"{split}-groups.jsonl"
        tick = time.perf_counter()
        with records_path.open("w", encoding="utf-8") as handle:
            def raw_rows():
                # Arrow converts request batches efficiently; no neural block
                # decoding or per-row Arrow table conversion is necessary.
                for start in range(0, len(indexes), 1024):
                    yield from dataset.rows.take(pa.array(indexes[start:start+1024])).to_pylist()
            for number, (row, size) in enumerate(zip(raw_rows(), lengths), 1):
                end = offset + int(size)
                x[offset:end] = builder(row)
                y[offset:end] = row["labels"]
                weight = (.5 / (train_search * int(size)) if row["task"] == "search"
                          else .5 / train_rec) if split == "train" else 1.
                # Match sklearn's C convention and the original baseline:
                # mean row weight 1, task totals each N/2.
                w[offset:end] = weight * count if split == "train" else weight
                summary = {key: row[key] for key in ("user_raw", "request_id", "task", "cohort")}
                summary.update(offset=offset, size=int(size))
                handle.write(json.dumps(summary) + "\n")
                offset = end
                if number % 25000 == 0:
                    print(json.dumps({"phase": "features", "split": split, "requests": number,
                        "total_requests": len(indexes), "elapsed_seconds": time.perf_counter() - tick}), flush=True)
        assert offset == count
        for array in (x, y, w):
            array.flush()
        del x, y, w
        info[split] = {"requests": len(indexes), "candidates": count,
                       "seconds": time.perf_counter() - tick}
        paths.extend([f"{split}-{suffix}.npy" for suffix in ("x", "y", "w")])
        paths.append(records_path.name)
    manifest = {"contract": contract, "splits": info, "seconds": time.perf_counter() - started,
                "files": {name: digest(output / name) for name in paths},
                "rss_bytes_observed": psutil.Process().memory_info().rss}
    write_json(manifest_path, manifest)
    return manifest


def score_array(dataset, scores, path):
    """Export exactly the original validation request/candidate order."""
    offset = 0
    with Path(path).open("w", encoding="utf-8") as handle:
        for index in _split_indexes(dataset, "validation"):
            row = dataset.rows.slice(int(index), 1).to_pylist()[0]
            size = len(row["item_raw"])
            values = np.asarray(scores[offset:offset + size]).tolist()
            common = {key: row[key] for key in ("task", "user_raw", "request_id", "cohort")}
            common["fallback"] = False
            records = ([{**common, "item_raw": row["item_raw"], "labels": row["labels"], "scores": values}]
                       if row["task"] == "search" else
                       [{**common, "item_raw": item, "label": label, "score": value}
                        for item, label, value in zip(row["item_raw"], row["labels"], values)])
            for record in records:
                handle.write(json.dumps(record, allow_nan=False) + "\n")
            offset += size
    if offset != len(scores):
        raise ValueError("Prediction count mismatch")
    return evaluate_predictions(jsonl(path))


def baselines(dataset, output, threads=4):
    """One fixed logistic setting, metadata and no-caption sensitivity."""
    output = Path(output)
    cache = output / "features"
    prepare_features(dataset, cache)
    x = np.load(cache / "train-x.npy", mmap_mode="r")
    y = np.load(cache / "train-y.npy", mmap_mode="r")
    w = np.load(cache / "train-w.npy", mmap_mode="r")
    vx = np.load(cache / "validation-x.npy", mmap_mode="r")
    results = {}
    for name, columns in (("logistic-metadata", list(range(17))), ("logistic-no-captions", list(range(10)))):
        path = output / name
        path.mkdir(parents=True, exist_ok=True)
        if (path / "result.json").exists():
            results[name] = read_json(path / "result.json")
            continue
        tick = time.perf_counter()
        with threadpool_limits(limits=threads):
            # Bounded dimensionality (17/10), no whole-cohort Python histories.
            features = np.empty((len(x), len(columns)), dtype=np.float64, order="C")
            # Copy in chunks to avoid a second full Fortran temporary from
            # NumPy advanced column indexing on a memory-constrained laptop.
            for start in range(0, len(x), 32768):
                features[start:start+32768] = x[start:start+32768, columns]
            scaler = StandardScaler().fit(features, sample_weight=w)
            scaler.transform(features, copy=False)
            classifier = LogisticRegression(C=1., max_iter=300, random_state=42)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                classifier.fit(features, y, sample_weight=w)
            convergence = [str(item.message) for item in caught if issubclass(item.category, ConvergenceWarning)]
            del features
            predicted = classifier.decision_function(scaler.transform(vx[:, columns]))
        metrics = score_array(dataset, predicted, path / "validation.jsonl")
        result = {"model": name, "data_hash": dataset.data_hash, "feature_policy": FEATURE_VERSION,
            "metrics": metrics, "fit_and_score_seconds": time.perf_counter() - tick,
            "n_iter": classifier.n_iter_.tolist(), "converged": not bool(convergence),
            "convergence_warnings": convergence, "test_accessed": False,
            "train_candidates": len(y), "validation_candidates": len(predicted),
            "prediction_sha256": digest(path / "validation.jsonl"),
            "rss_bytes_observed": psutil.Process().memory_info().rss}
        write_json(path / "model.json", {"format": "trace-safe-logistic-json-v1", "columns": columns,
            "feature_names": (SafeFeatureExtractor().feature_names + EXTRA_FEATURES),
            "coef": classifier.coef_.tolist(), "intercept": classifier.intercept_.tolist(),
            "classes": classifier.classes_.tolist(), "scale": scaler.scale_.tolist(), "mean": scaler.mean_.tolist(),
            "frequency_state": "use train-only frequency arrays in original prepared data"})
        write_json(path / "result.json", result)
        results[name] = result
        print(json.dumps({"phase": name, "converged": result["converged"], "seconds": result["fit_and_score_seconds"]}), flush=True)
    # Simple fixed rule: train frequency + raw query-caption Jaccard, plus a
    # train-only per-task intercept. No per-item label aggregates.
    name = "frequency-raw-overlap"
    path = output / name
    path.mkdir(parents=True, exist_ok=True)
    intercepts = {}
    train_task = np.empty(len(y), dtype=bool)
    for group in jsonl(cache / "train-groups.jsonl"):
        a, n = group["offset"], group["size"]
        train_task[a:a+n] = group["task"] == "search"
    for task, mask in (("search", train_task), ("recommendation", ~train_task)):
        prior = float(np.average(y[mask], weights=w[mask]))
        prior = min(1-1e-6, max(1e-6, prior))
        intercepts[task] = math.log(prior / (1-prior))
    predicted = vx[:, 0] / max(1., dataset.frequency_scale) + vx[:, 16]
    for group in jsonl(cache / "validation-groups.jsonl"):
        a, n = group["offset"], group["size"]
        predicted[a:a+n] += intercepts[group["task"]]
    metrics = score_array(dataset, predicted, path / "validation.jsonl")
    results[name] = {"model": name, "metrics": metrics, "data_hash": dataset.data_hash,
        "task_intercepts": intercepts, "test_accessed": False,
        "prediction_sha256": digest(path / "validation.jsonl")}
    write_json(path / "result.json", results[name])
    write_json(output / "baseline-results.json", results)
    return results


def identity(row):
    if row["task"] == "search":
        return ("search", str(row["user_raw"]), str(row["request_id"]))
    return ("recommendation", str(row["user_raw"]), str(row["request_id"]), str(row["item_raw"]))


def branch_diagnostics(model, dataset, config, saved_predictions, output, threads=4):
    """Full validation CPU parity and fixed-weight interventions; no fitting."""
    import torch
    from src.models.trace_v2 import collate_v2, group_bce
    from src.training.trace_v2 import partition
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(threads)
    model.eval()
    expected = {}
    for row in jsonl(saved_predictions):
        key = identity(row)
        if key in expected:
            raise ValueError(f"Duplicate saved prediction identity: {key}")
        expected[key] = row
    selected_metrics = evaluate_predictions(expected.values())
    handles = {b: (output / f"{b}.jsonl").open("w", encoding="utf-8") for b in BRANCHES}
    diag = (output / "requests.jsonl").open("w", encoding="utf-8")
    processed = set()
    max_error, candidate_count = 0., 0
    abs_errors, gates, ticks = [], defaultdict(list), []
    start = time.perf_counter()
    try:
        with torch.inference_mode():
            for task in ("search", "recommendation"):
                indexes = dataset.indexes[("validation", task)]
                for block in range(0, len(indexes), 32):
                    rows = [dataset.encode(int(i)) for i in indexes[block:block+32]]
                    limit = config["recommendation_microbatch"] if task == "recommendation" else config["microbatch_requests"]
                    for part in partition(rows, limit, config["candidate_budget"]):
                        tick = time.perf_counter()
                        batch = collate_v2(part, config["model"]["max_history"])
                        out = model(batch, config["candidate_chunk"])
                        ticks.append(time.perf_counter() - tick)
                        losses = {b: group_bce(out[b], batch).tolist() for b in BRANCHES}
                        scores = {b: out[b].tolist() for b in BRANCHES}
                        offset = 0
                        for j, row in enumerate(part):
                            n = len(row["item_raw"])
                            common = {key: row[key] for key in ("task", "user_raw", "request_id", "cohort")}
                            common["fallback"] = False
                            reference = None
                            for b in BRANCHES:
                                values = scores[b][offset:offset+n]
                                records = ([{**common, "item_raw": row["item_raw"], "labels": row["labels"], "scores": values}]
                                           if task == "search" else
                                           [{**common, "item_raw": item, "label": label, "score": value}
                                            for item, label, value in zip(row["item_raw"], row["labels"], values)])
                                for record in records:
                                    handles[b].write(json.dumps(record, allow_nan=False) + "\n")
                                    if b != "final":
                                        continue
                                    key = identity(record)
                                    if key in processed or key not in expected:
                                        raise ValueError(f"Unexpected/duplicate validation identity: {key}")
                                    ref = expected[key]
                                    if task == "search":
                                        if record["item_raw"] != ref["item_raw"] or record["labels"] != ref["labels"]:
                                            raise ValueError("Saved search candidate/label alignment mismatch")
                                        a, z = np.asarray(record["scores"]), np.asarray(ref["scores"])
                                        reference = ref
                                    else:
                                        if record["label"] != ref["label"]:
                                            raise ValueError("Saved recommendation label mismatch")
                                        a, z = np.asarray([record["score"]]), np.asarray([ref["score"]])
                                    error = np.abs(a-z)
                                    max_error = max(max_error, float(error.max()))
                                    candidate_count += len(error)
                                    abs_errors.append(float(error.sum()))
                                    # Declared CPU (torch 2.5) vs Kaggle GPU
                                    # (torch 2.11) tolerance, never silently widen.
                                    np.testing.assert_allclose(a, z, atol=2e-4, rtol=2e-4)
                                    processed.add(key)
                            gate = float(out["gate"][j])
                            cross = bool(out["scalars"][j, 1])
                            same = bool(out["scalars"][j, 0])
                            if cross:
                                gates[task].append(gate)
                            diagnostic = {**common, "gate": gate, "cross_history": cross,
                                "same_history": same, "history_blocks": len(row["history"]),
                                "oov_fraction": sum(i == 1 for i in row["item_ids"]) / n,
                                "query_present": bool(row["query_ids"]),
                                "query_known_fraction": sum(i > 1 for i in row["query_ids"]) / max(1, len(row["query_ids"])),
                                "max_raw_query_caption_overlap": max(f[0] for f in row["relevance_features"]),
                                "candidates": n, "loss": {b: losses[b][j] for b in BRANCHES},
                                "item_raw": row["item_raw"], "labels": row["labels"],
                                "item_known": [i > 1 for i in row["item_ids"]],
                                "mean_abs_cross_residual": float(out["residual"][offset:offset+n].abs().mean())}
                            if task == "search":
                                diagnostic["ndcg"] = {b: search_request_metrics({**common, "item_raw": row["item_raw"],
                                    "labels": row["labels"], "scores": scores[b][offset:offset+n]})["ndcg"]
                                    for b in BRANCHES} if reference else {}
                            diag.write(json.dumps(diagnostic, allow_nan=False) + "\n")
                            offset += n
                    if block % 3200 == 0:
                        print(json.dumps({"phase": "branches", "task": task, "requests": min(block+32, len(indexes)),
                            "total_requests": len(indexes), "elapsed_seconds": time.perf_counter()-start}), flush=True)
    finally:
        for handle in handles.values():
            handle.close()
        diag.close()
    if processed != set(expected):
        raise ValueError("Saved validation identities were not fully reproduced")
    metrics = {b: evaluate_predictions(jsonl(output / f"{b}.jsonl")) for b in BRANCHES}
    metric_delta = {
        "search_user_ndcg_at_10": metrics["final"]["search"]["overall"]["user_macro"]["ndcg@10"]
            - selected_metrics["search"]["overall"]["user_macro"]["ndcg@10"],
        "recommendation_ap": metrics["final"]["recommendation"]["average_precision"]
            - selected_metrics["recommendation"]["average_precision"],
    }
    result = {"data_hash": dataset.data_hash, "test_accessed": False,
        "intervention": "frozen weights, not retrained ablations", "metrics": metrics,
        "saved_gpu_metrics": selected_metrics, "parity": {"passed": True, "records": len(processed),
            "candidates": candidate_count, "max_abs_logit_error": max_error,
            "mean_abs_logit_error": math.fsum(abs_errors)/candidate_count, "atol": 2e-4, "rtol": 2e-4,
            "cpu_minus_saved_gpu_metric_delta": metric_delta,
            "rank_note": "Small logit differences can reorder near ties; metric parity is reported separately."},
        "gate_cross_history_only": {task: {"count": len(v), "mean": float(np.mean(v)),
            "percentiles_5_50_95": np.percentile(v, [5,50,95]).tolist(),
            "fraction_below_005": float(np.mean(np.asarray(v)<.05)),
            "fraction_above_095": float(np.mean(np.asarray(v)>.95))} for task,v in gates.items()},
        "inference_seconds": time.perf_counter()-start,
        "microbatch_latency_mean_seconds": float(np.mean(ticks)),
        "prediction_sha256": {b: digest(output / f"{b}.jsonl") for b in BRANCHES}}
    write_json(output / "branch-results.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--phase", choices=("features", "baselines", "branches"), required=True)
    p.add_argument("--weights", type=Path)
    p.add_argument("--saved-predictions", type=Path)
    p.add_argument("--config", type=Path)
    p.add_argument("--frequency-state", type=Path)
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    dataset = V2Dataset(args.data)
    if args.phase == "features":
        prepare_features(dataset, args.output / "features")
    elif args.phase == "baselines":
        baselines(dataset, args.output / "baselines", args.threads)
    else:
        from src.training.trace_v2 import load_export
        config = read_json(args.config)
        config = config.get("config", config)
        if dataset.data_hash != config["prepared_data_hash"]:
            raise ValueError("Model dataset mismatch")
        import torch
        value = torch.load(args.weights, map_location="cpu", weights_only=True, mmap=True)
        if (value["data_hash"] != dataset.data_hash or value["architecture"] != config["model"]
                or value["item_count"] != dataset.item_count or value["token_count"] != dataset.token_count):
            raise ValueError("Inference export architecture/vocabulary/data mismatch")
        del value
        if args.frequency_state:
            state = read_json(args.frequency_state)
            np.testing.assert_array_equal(state["frequency_ids"], dataset.frequency_ids)
            np.testing.assert_array_equal(state["frequencies"], dataset.frequencies)
            assert state["frequency_scale"] == dataset.frequency_scale
        model = load_export(args.weights)
        branch_diagnostics(model, dataset, config, args.saved_predictions, args.output / "branches", args.threads)


if __name__ == "__main__":
    main()
