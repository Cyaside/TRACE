"""Arrow-backed V2 requests. Decode sampled batches, never expand all histories.

Vocabulary/manifest are reused without modification. Test outcomes are filtered
out at the Parquet reader. Released caption timing remains a snapshot assumption.
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from src.common import digest, read_json


class V2Dataset:
    def __init__(self, root, cache_size=2048, verify=True):
        self.root = Path(root)
        self.manifest = read_json(self.root / "manifest.json")
        if verify:
            for name, expected in self.manifest["files"].items():
                if digest(self.root / name) != expected:
                    raise ValueError(f"Prepared hash mismatch: {name}")
        self.data_hash = self.manifest["data_hash"]
        self.item_count = self.manifest["item_count"]
        self.token_count = self.manifest["token_count"]
        vocab = read_json(self.root / "vocabulary.json")
        self.items = np.asarray(vocab["items"], dtype=np.int64)
        self.tokens = np.asarray(vocab["tokens"], dtype=np.int64)
        del vocab
        if np.any(self.items[1:] <= self.items[:-1]) or np.any(self.tokens[1:] <= self.tokens[:-1]):
            raise ValueError("Vocabulary must have unique sorted raw IDs.")
        self.rows = pq.read_table(self.root / "requests.parquet",
                                  filters=[("split", "in", ["train", "validation"])])
        self.blocks = pq.read_table(self.root / "blocks.parquet",
                                   columns=["timestamp_ms", "rec_items", "search_queries", "service_mask"])
        self.metadata = pq.read_table(self.root / "metadata.parquet", columns=["item_raw", "caption_tokens"])
        self.meta_ids = self.metadata["item_raw"].combine_chunks().to_numpy()
        self.meta_order = np.argsort(self.meta_ids)
        self.meta_ids = self.meta_ids[self.meta_order]
        split = self.rows["split"].to_numpy()
        task = self.rows["task"].to_numpy()
        self.indexes = {(s, t): np.flatnonzero((split == s) & (task == t))
                        for s in ("train", "validation") for t in ("search", "recommendation")}
        train = self.rows.filter(pc.equal(self.rows["split"], "train"))
        # Count exposures, never use click labels to fit the relevance feature.
        flat = pc.list_flatten(train["item_raw"]).combine_chunks().to_numpy()
        raw, counts = np.unique(flat, return_counts=True)
        self.frequency_ids, self.frequencies = raw, counts
        self.frequency_scale = float(np.log1p(counts.max()))
        rec = self.rows.take(pa.array(self.indexes[("train", "recommendation")]))
        self.rec_offsets = np.cumsum(pc.list_value_length(rec["item_raw"]).to_numpy(), dtype=np.int64)
        self.rec_count = int(self.rec_offsets[-1])
        self.cache_size = int(cache_size)
        self.cache = OrderedDict()

    @staticmethod
    def lookup(vocab, values):
        values = np.asarray(values, dtype=np.int64)
        positions = np.searchsorted(vocab, values)
        valid = positions < len(vocab)
        matched = np.zeros(len(values), dtype=bool)
        matched[valid] = vocab[positions[valid]] == values[valid]
        return np.where(matched, positions + 2, 1).tolist()

    def token_ids(self, values):
        return self.lookup(self.tokens, values[:50])

    def block(self, index):
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index]
        row = self.blocks.slice(index, 1).to_pylist()[0]
        value = {"timestamp_ms": row["timestamp_ms"], "service_mask": row["service_mask"],
                 "rec_item_ids": self.lookup(self.items, row["rec_items"]),
                 "search_query_ids": [self.token_ids(q) for q in row["search_queries"]]}
        self.cache[index] = value
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return value

    def encode(self, index, column=None):
        row = self.rows.slice(int(index), 1).to_pylist()[0]
        if column is not None:
            for key in ("item_raw", "labels"):
                row[key] = [row[key][int(column)]]
        captions, features = [], []
        query = set(row["query_tokens"][:50]) if row["task"] == "search" else set()
        for raw in row["item_raw"]:
            position = int(np.searchsorted(self.meta_ids, raw))
            caption = []
            if position < len(self.meta_ids) and self.meta_ids[position] == raw:
                caption = self.metadata["caption_tokens"][int(self.meta_order[position])].as_py()[:50]
            captions.append(self.token_ids(caption))
            words = set(caption)
            overlap = len(query & words)
            freq_position = int(np.searchsorted(self.frequency_ids, raw))
            freq = (int(self.frequencies[freq_position]) if freq_position < len(self.frequency_ids)
                    and self.frequency_ids[freq_position] == raw else 0)
            features.append([overlap / max(1, len(query)), overlap / max(1, len(words)),
                             overlap / max(1, len(query | words)), np.log1p(freq) / self.frequency_scale])
        return {**row, "item_ids": self.lookup(self.items, row["item_raw"]),
                "caption_ids": captions, "query_ids": self.token_ids(row["query_tokens"]),
                "relevance_features": features,
                "history": [self.block(i) for i in row["history_ids"][-50:]]}

    def iter_requests(self, split, task):
        if split not in ("train", "validation"):
            raise ValueError("This training reader excludes test outcomes.")
        for index in self.indexes[(split, task)]:
            yield self.encode(index)

    def feature_state(self):
        return {"frequency_ids": self.frequency_ids.tolist(), "frequencies": self.frequencies.tolist(),
                "frequency_scale": self.frequency_scale,
                "normalization": "fixed log1p(count)/log1p(50), log1p(clipped_gap_seconds)/log1p(1e6)"}


class V2Sampler:
    """Uniform complete search requests and uniform individual rec exposures."""
    def __init__(self, dataset, seed):
        import random
        self.dataset, self.rng = dataset, random.Random(seed)
        self.cursor = 0

    def batch(self, task, size):
        dataset = self.dataset
        indexes = dataset.indexes[("train", task)]
        if task == "search":
            result = [dataset.encode(indexes[self.rng.randrange(len(indexes))]) for _ in range(size)]
        elif task == "recommendation":
            result = []
            for _ in range(size):
                exposure = self.rng.randrange(dataset.rec_count)
                row = int(np.searchsorted(dataset.rec_offsets, exposure, side="right"))
                column = exposure - (int(dataset.rec_offsets[row - 1]) if row else 0)
                result.append(dataset.encode(indexes[row], column))
        else:
            raise ValueError("Unknown task")
        self.cursor += size
        return result

    def state_dict(self):
        return {"rng": self.rng.getstate(), "cursor": self.cursor}

    def load_state_dict(self, state):
        self.rng.setstate(state["rng"])
        self.cursor = state["cursor"]
