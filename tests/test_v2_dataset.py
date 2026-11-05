"""Arrow reader equivalence, train-only features and bounded lazy sampling."""
import json

import pyarrow as pa
import pyarrow.parquet as pq

from src.common import digest, object_hash, write_json
from src.features.dataset import PreparedDataset
from src.features.v2_dataset import V2Dataset, V2Sampler


def prepared(tmp_path):
    blocks = [{"timestamp_ms": 1000, "rec_items": [20, 30], "search_queries": [[100, 200]],
               "service_mask": 3, "gap_bucket": 0}]
    base = {"user_raw": 1, "timestamp_ms": 2000, "item_raw": [20, 30], "labels": [1, 0],
            "query_tokens": [100], "history_ids": [0], "request_gap": 1, "cohort": "transition"}
    rows = [{**base, "request_id": "a", "split": "train", "task": "search"},
            {**base, "request_id": "b", "split": "train", "task": "recommendation"},
            {**base, "request_id": "v", "split": "validation", "task": "search", "item_raw": [99, 20]},
            {**base, "request_id": "w", "split": "validation", "task": "recommendation"},
            {**base, "request_id": "t", "split": "test", "task": "search", "labels": [0, 1]}]
    metadata = [{"item_raw": 99, "caption_tokens": [100, 999]},
                {"item_raw": 30, "caption_tokens": [200]}, {"item_raw": 20, "caption_tokens": [100]}]
    for name, values in (("requests", rows), ("blocks", blocks), ("metadata", metadata)):
        pq.write_table(pa.Table.from_pylist(values), tmp_path / f"{name}.parquet")
    write_json(tmp_path / "vocabulary.json", {"items": [20, 30], "tokens": [100, 200]})
    files = {name: digest(tmp_path / name) for name in ("requests.parquet", "blocks.parquet", "metadata.parquet", "vocabulary.json")}
    write_json(tmp_path / "manifest.json", {"files": files, "data_hash": object_hash(files), "item_count": 4, "token_count": 4})
    return tmp_path


def test_lazy_reader_matches_existing_ids_and_excludes_test(tmp_path):
    root = prepared(tmp_path)
    old, new = PreparedDataset(root), V2Dataset(root, cache_size=1)
    assert len(new.rows) == 4
    assert new.rows["split"].to_pylist() == ["train", "train", "validation", "validation"]
    for task in ("search", "recommendation"):
        a = old.requests("train", task)[0]
        b = next(new.iter_requests("train", task))
        for key in ("item_ids", "caption_ids", "query_ids", "labels", "item_raw"):
            assert a[key] == b[key]
        for x, y in zip(a["history"], b["history"]):
            for key in ("rec_item_ids", "search_query_ids", "timestamp_ms", "service_mask"):
                assert x[key] == y[key]
    unseen = next(new.iter_requests("validation", "search"))
    assert unseen["item_ids"][0] == 1
    assert unseen["caption_ids"][0] == [2, 1]
    assert unseen["relevance_features"][0][0] == 1  # raw matching survives OOV collapse
    assert unseen["relevance_features"][0][3] == 0  # val-only item has zero train frequency
    assert len(new.cache) <= 1


def test_exposure_sampler_resumes_and_selects_all_aligned_fields(tmp_path):
    dataset = V2Dataset(prepared(tmp_path))
    sampler = V2Sampler(dataset, 42)
    sampler.batch("search", 3)
    state = sampler.state_dict()
    first = sampler.batch("recommendation", 20)
    sampler.load_state_dict(state)
    second = sampler.batch("recommendation", 20)
    assert first == second
    assert {row["item_raw"][0] for row in first} == {20, 30}
    for row in first:
        for key in ("item_raw", "item_ids", "labels", "caption_ids", "relevance_features"):
            assert len(row[key]) == 1
