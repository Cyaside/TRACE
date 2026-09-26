"""Independent, bounded read-only TRACE data gate verification.

Run after resource profiling, since ZIP/CRC verification reads about 2.6 GB.
Only --output is written. No training or model fitting occurs.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
import zipfile
import zlib

import duckdb
import pyarrow.parquet as pq


def file_checksums(path):
    sha = hashlib.sha256()
    md5 = hashlib.md5()
    crc = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            sha.update(chunk)
            md5.update(chunk)
            crc = zlib.crc32(chunk, crc)
    return {"bytes": path.stat().st_size, "sha256": sha.hexdigest(),
            "md5": md5.hexdigest(), "crc32": crc & 0xffffffff}


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def literal(path):
    return "'" + path.as_posix().replace("'", "''") + "'"


def verify(root, output):
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    evidence = {"root": str(root.resolve()), "checks": {}}
    checks = evidence["checks"]
    release = read(root / "data/raw/raw-release.json")
    archive_path = root / "data/raw/KuaiSAR.zip"
    archive = file_checksums(archive_path)
    assert archive["md5"] == "daea8cbf605db6bd5841740f0e4a12d9" == release["md5"]
    assert archive["sha256"] == release["sha256"]
    assert archive["bytes"] == release["bytes"]
    checks["archive_integrity"] = archive
    print("Verified archive MD5/SHA256", flush=True)
    extracted = {}
    with zipfile.ZipFile(archive_path) as archive_file:
        for info in archive_file.infolist():
            if info.is_dir():
                continue
            name = Path(info.filename).name
            result = file_checksums(root / "data/raw/extracted" / name)
            assert result["bytes"] == info.file_size
            assert result["crc32"] == info.CRC
            extracted[name] = result
            print(f"Verified extracted size/CRC: {name}", flush=True)
    checks["extracted_payloads"] = extracted

    audit = read(root / "data/canonical/data-audit.json")
    db = duckdb.connect(str(root / "data/canonical/audit.duckdb"), read_only=True)
    db.execute("SET threads=4")
    db.execute("SET memory_limit='2GB'")
    db.execute("SET preserve_insertion_order=true")
    scalar = lambda sql: db.execute(sql).fetchone()[0]
    users = [row[0] for row in db.execute("SELECT user_raw FROM eligible").fetchall()]
    users.sort(key=lambda user: hashlib.sha256(f"42:{user}".encode("utf-8")).hexdigest())
    assert len(users) == audit["eligible_users"]
    cutoff = audit["cutoffs_utc8_ms"][0]
    eligibility_expected = set(row[0] for row in db.execute(f"""SELECT DISTINCT g.user_raw
        FROM search_groups g WHERE NOT g.conflict AND g.timestamp_ms < {cutoff}
        AND EXISTS(SELECT 1 FROM search_kept k WHERE k.user_raw=g.user_raw AND k.session_raw=g.session_raw)
        AND EXISTS(SELECT 1 FROM rec r WHERE r.user_raw=g.user_raw AND r.timestamp_ms < {cutoff})""").fetchall())
    assert set(users) == eligibility_expected
    smoke_users = read(root / "data/canonical/smoke/users.json")
    pilot_users = read(root / "data/canonical/pilot/users.json")
    assert smoke_users == users[:200] and pilot_users == users[:2000]
    assert smoke_users == pilot_users[:200]
    checks["deterministic_cohorts"] = {"eligible_users": len(users), "smoke": len(smoke_users),
        "pilot": len(pilot_users), "nested": True, "rule": "sha256 UTF8 42:<numeric raw user ID>"}
    assert scalar("SELECT count(*) FROM rec") == audit["raw_rows"]["rec_inter.csv"]
    assert scalar("SELECT count(*) FROM search_raw") == audit["raw_rows"]["src_inter.csv"]
    assert scalar("SELECT count(*) FROM search_groups WHERE conflict") == audit["quarantined_groups"]
    assert scalar("SELECT coalesce(sum(raw_count),0) FROM search_groups WHERE conflict") == audit["quarantined_rows"]
    assert scalar("SELECT count(*) FROM rec WHERE label NOT IN (0,1) OR label IS NULL") == 0
    assert scalar("SELECT count(*) FROM search_kept WHERE label NOT IN (0,1) OR label IS NULL") == 0
    assert scalar("SELECT count(*) FROM (SELECT user_raw,session_raw,item_raw,count(*) c FROM search_kept GROUP BY ALL HAVING c>1)") == 0
    assert scalar("SELECT count(*) FROM metadata") == scalar("SELECT count(DISTINCT item_raw) FROM metadata")
    duplicate_rec = scalar("SELECT count(*) FROM rec") - scalar("SELECT count(*) FROM (SELECT DISTINCT user_raw,timestamp_ms,item_raw FROM rec)")
    assert duplicate_rec == audit["recommendation_duplicate_exposures"]
    checks["raw_projection"] = {"search_rows": audit["raw_rows"]["src_inter.csv"],
        "recommendation_rows": audit["raw_rows"]["rec_inter.csv"],
        "quarantined_groups": audit["quarantined_groups"], "quarantined_rows": audit["quarantined_rows"],
        "recommendation_duplicate_exposures": duplicate_rec, "binary_labels": True, "unique_search_candidates": True}
    exact_raw_numbers = {
        "recommendation_invalid_spellings": scalar(r"""SELECT count(*) FROM rec_inter WHERE
            NOT regexp_full_match(coalesce(user_id,''), '[0-9]+') OR
            NOT regexp_full_match(coalesce(item_id,''), '[0-9]+') OR
            NOT regexp_full_match(coalesce(timestamp,''), '[0-9]+(?:\.0+)?') OR
            click IS NULL OR NOT regexp_full_match(click, '(?:0|1)(?:\.0+)?')"""),
        "search_invalid_spellings": scalar(r"""SELECT count(*) FROM src_inter WHERE
            NOT regexp_full_match(coalesce(user_id,''), '[0-9]+') OR
            NOT regexp_full_match(coalesce(item_id,''), '[0-9]+') OR
            NOT regexp_full_match(coalesce(search_session_id,''), '[0-9]+') OR
            NOT regexp_full_match(coalesce(search_session_timestamp,''), '[0-9]+(?:\.0+)?') OR
            NOT regexp_full_match(coalesce(click_cnt,''), '[0-9]+(?:\.0+)?')"""),
        "metadata_key_invalid_spellings": scalar("SELECT count(*) FROM item_features WHERE NOT regexp_full_match(coalesce(item_id,''),'[0-9]+')"),
    }
    assert not any(exact_raw_numbers.values())
    checks["exact_raw_number_spellings"] = exact_raw_numbers

    cohorts = {}
    for name, selected_users in (("smoke", smoke_users), ("pilot", pilot_users)):
        canonical = root / "data/canonical" / name
        manifest = read(canonical / "manifest.json")
        assert manifest["data_hash"] == object_hash({key: value for key, value in manifest.items() if key != "data_hash"})
        assert manifest["cohort_hash"] == object_hash(selected_users)
        verified_files = {}
        for filename, expected in manifest["files"].items():
            actual = file_checksums(canonical / filename)["sha256"]
            assert actual == expected
            verified_files[filename] = actual
        derived = output / name
        derived.mkdir(exist_ok=True)
        cohort_sql = "(SELECT unnest([" + ",".join(map(str, selected_users)) + "]) user_raw) cohort_users"
        copied = {}
        for table, filename, order in (("selected_search", "search.parquet", "user_raw,timestamp_ms,session_raw"),
                                       ("selected_rec", "recommendation.parquet", "user_raw,timestamp_ms,item_raw,raw_row")):
            path = derived / filename
            db.execute(f"COPY (SELECT t.* FROM {table} t JOIN {cohort_sql} USING(user_raw) ORDER BY {order}) TO {literal(path)} (FORMAT PARQUET, COMPRESSION ZSTD)")
            copied[filename] = {"sha256": file_checksums(path)["sha256"], "expected": manifest["files"][filename]}
            copied[filename]["byte_identical"] = copied[filename]["sha256"] == copied[filename]["expected"]
            if not copied[filename]["byte_identical"]:
                identical = scalar(f"SELECT count(*) FROM ((SELECT * FROM read_parquet({literal(path)}) EXCEPT ALL SELECT * FROM read_parquet({literal(canonical/filename)})) UNION ALL (SELECT * FROM read_parquet({literal(canonical/filename)}) EXCEPT ALL SELECT * FROM read_parquet({literal(path)})))") == 0
                copied[filename]["rows_identical"] = identical
                assert identical
        metadata_path = derived / "metadata.parquet"
        db.execute(f"COPY (SELECT * FROM metadata ORDER BY item_raw) TO {literal(metadata_path)} (FORMAT PARQUET, COMPRESSION ZSTD)")
        copied["metadata.parquet"] = {"sha256": file_checksums(metadata_path)["sha256"], "expected": manifest["files"]["metadata.parquet"]}
        copied["metadata.parquet"]["byte_identical"] = copied["metadata.parquet"]["sha256"] == copied["metadata.parquet"]["expected"]
        if not copied["metadata.parquet"]["byte_identical"]:
            identical = scalar(f"SELECT count(*) FROM ((SELECT * FROM read_parquet({literal(metadata_path)}) EXCEPT ALL SELECT * FROM read_parquet({literal(canonical/'metadata.parquet')})) UNION ALL (SELECT * FROM read_parquet({literal(canonical/'metadata.parquet')}) EXCEPT ALL SELECT * FROM read_parquet({literal(metadata_path)})))") == 0
            copied["metadata.parquet"]["rows_identical"] = identical
            assert identical
        counts = manifest["counts"]
        assert scalar(f"SELECT count(*) FROM selected_search JOIN {cohort_sql} USING(user_raw)") == counts["search_groups"]
        assert scalar(f"SELECT count(*) FROM selected_rec JOIN {cohort_sql} USING(user_raw)") == counts["recommendation_exposures"]
        assert scalar(f"SELECT sum(len(item_raw)) FROM selected_search JOIN {cohort_sql} USING(user_raw)") == counts["retained_search_candidates"]
        assert scalar(f"SELECT sum(raw_count) FROM selected_search JOIN {cohort_sql} USING(user_raw)") == counts["original_search_rows"]
        assert scalar(f"SELECT sum(retained_raw_count) FROM selected_search JOIN {cohort_sql} USING(user_raw)") == counts["retained_search_rows_before_dedup"]
        original_positive = scalar(f"SELECT sum(cast(r.label AS BIGINT)) FROM search_raw r JOIN {cohort_sql} USING(user_raw) JOIN search_groups g USING(user_raw,session_raw) WHERE NOT g.conflict")
        retained_positive = scalar(f"SELECT sum(cast(r.label AS BIGINT)) FROM search_raw r JOIN {cohort_sql} USING(user_raw) JOIN search_groups g USING(user_raw,session_raw) WHERE NOT g.conflict AND r.item_type IN ('VIDEO','IMAGE_ATLAS')")
        assert original_positive == counts["original_search_positive_rows"]
        assert retained_positive == counts["retained_search_positive_rows"]
        missing = scalar(f"SELECT count(*) FROM (SELECT unnest(item_raw) item_raw FROM selected_search JOIN {cohort_sql} USING(user_raw)) c LEFT JOIN metadata m USING(item_raw) WHERE m.item_raw IS NULL")
        assert missing == counts["missing_search_metadata_candidates"]
        cohorts[name] = {"verified_files": verified_files, "reproduced_copy": copied, "counts": counts,
                        "positive_retention_fraction": counts["retained_search_positive_rows"] / counts["original_search_positive_rows"],
                        "retained_unique_positive_candidates": scalar(f"SELECT sum(label) FROM (SELECT unnest(labels) AS label FROM selected_search JOIN {cohort_sql} USING(user_raw))"),
                        "retained_missing_metadata_candidates": missing}
        print(f"Verified canonical {name} selections/checksums/counts", flush=True)
    checks["canonical"] = cohorts
    db.close()

    prepared = {}
    for name in ("smoke", "pilot"):
        path = root / "data/prepared" / (name + "-v1")
        manifest = read(path / "manifest.json")
        canonical_manifest = read(root / "data/canonical" / name / "manifest.json")
        assert manifest["canonical_hash"] == canonical_manifest["data_hash"], "Prepared canonical lineage mismatch"
        assert manifest["cohort_hash"] == canonical_manifest["cohort_hash"]
        assert manifest["data_hash"] == object_hash({key: value for key, value in manifest.items() if key != "data_hash"})
        for filename, expected in manifest["files"].items():
            assert file_checksums(path / filename)["sha256"] == expected
        requests = pq.read_table(path / "requests.parquet").to_pylist()
        blocks = pq.read_table(path / "blocks.parquet").to_pylist()
        vocab = read(path / "vocabulary.json")
        cutoffs = manifest["cutoffs_utc8_ms"]
        canonical = root / "data/canonical" / name
        search = pq.read_table(canonical / "search.parquet").to_pylist()
        rec = pq.read_table(canonical / "recommendation.parquet").to_pylist()
        train_items = {item for row in search if row["timestamp_ms"] < cutoffs[0] for item in row["item_raw"]}
        train_items.update(row["item_raw"] for row in rec if row["timestamp_ms"] < cutoffs[0])
        train_tokens = {token for row in search if row["timestamp_ms"] < cutoffs[0] for token in json.loads(row["query_signature"])}
        metadata = pq.read_table(canonical / "metadata.parquet").to_pylist()
        for row in metadata:
            if row["item_raw"] in train_items:
                train_tokens.update(json.loads(row["caption"]) if row["caption"] is not None else [])
        assert vocab["items"] == sorted(train_items)
        assert vocab["tokens"] == sorted(train_tokens)
        assert vocab["pad"] == 0 and vocab["oov"] == 1
        assert manifest["item_count"] == len(train_items) + 2
        assert manifest["token_count"] == len(train_tokens) + 2
        assert object_hash(vocab) == manifest["vocabulary_hash"]
        for request in requests:
            expected_split = ("train" if request["timestamp_ms"] < cutoffs[0] else
                              "validation" if request["timestamp_ms"] < cutoffs[1] else
                              "test" if request["timestamp_ms"] < cutoffs[2] else "unused")
            assert request["split"] == expected_split != "unused"
            history = [blocks[index] for index in request["history_ids"]]
            assert len(history) <= 50
            assert all(block["user_raw"] == request["user_raw"] and block["timestamp_ms"] < request["timestamp_ms"] for block in history)
            assert all(left["timestamp_ms"] < right["timestamp_ms"] for left, right in zip(history, history[1:]))
            assert len(request["item_raw"]) == len(request["labels"])
            assert all(label in (0, 1) for label in request["labels"])
            if request["task"] == "search":
                assert len(set(request["item_raw"])) == len(request["item_raw"])
        assert all("labels" not in block and "search_result_ids" not in block for block in blocks)
        validation_transition = [row for row in requests if row["split"] == "validation" and row["task"] == "search" and row["cohort"] == "transition"]
        eligible_transition = [row for row in validation_transition if len(row["labels"]) >= 2 and 0 < sum(row["labels"]) < len(row["labels"])]
        prepared[name] = {"requests": len(requests), "blocks": len(blocks), "strict_prefixes_checked": len(requests),
                          "canonical_hash": manifest["canonical_hash"], "lineage_verified": True,
                          "training_items": len(train_items), "training_tokens": len(train_tokens),
                          "vocabulary_independently_reconstructed": True, "all_hashes_match": True,
                          "validation_transition_requests": len(validation_transition),
                          "validation_eligible_transition_requests": len(eligible_transition),
                          "validation_eligible_transition_users": len({row["user_raw"] for row in eligible_transition})}
        print(f"Verified prepared {name} strict prefixes, boundaries and vocabularies", flush=True)
    checks["prepared"] = prepared
    evidence["elapsed_seconds"] = time.perf_counter() - started
    evidence["status"] = "verified"
    (output / "verification.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": evidence["status"], "elapsed_seconds": evidence["elapsed_seconds"], "output": str(output)}, indent=2))
    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify(args.root, args.output)
