# Dataset acquisition and rebuild

TRACE downloads **KuaiSAR-small** from the authors' [Zenodo release](https://zenodo.org/records/8181109), then builds its own canonical cohorts and temporal features. Dataset payloads under `raw/`, `canonical/` and `prepared/` are generated locally and ignored by Git. Cloning TRACE does not download them.

## Pinned source

The [official project](https://kuaisar.github.io/) identifies `KuaiSAR.zip` as the small release; `KuaiSAR_v2.zip` is a different, larger release and is not accepted by this adapter.

| Identity | Expected value |
|---|---|
| URL | `https://zenodo.org/records/8181109/files/KuaiSAR.zip?download=1` |
| Size | `488077651` bytes, approximately 488 MB |
| MD5 published by Zenodo | `daea8cbf605db6bd5841740f0e4a12d9` |
| SHA-256 of the pinned archive | `ed8afa12196cbf18a719511a03b0915522c4039a6c08ce306df69aaeddb9fa1c` |

`src.data.download` checks the size and MD5 before accepting the archive, resumes interrupted downloads where the server supports it, and prints its SHA-256. Acquisition metadata is saved locally in `raw/raw-release.json`.

## Rebuild from a fresh clone

Run these commands from the repository root after completing the Python setup in the root README. They run data preparation on CPU; they do not train models or require a GPU.

```powershell
# 1. Download and verify the public archive.
.\.venv\Scripts\python.exe -m src.data.download

# 2. Extract, audit and create both canonical cohorts.
.\.venv\Scripts\python.exe -m src.data.audit

# 3. Create the small prepared cohort.
.\.venv\Scripts\python.exe -m src.features.prepare --canonical data/canonical/smoke --output data/prepared/smoke-v1

# 4. Create the pilot prepared cohort used by the baseline CLI defaults.
.\.venv\Scripts\python.exe -m src.features.prepare --canonical data/canonical/pilot --output data/prepared/pilot-v1
```

To independently check the archive SHA-256:

```powershell
Get-FileHash data/raw/KuaiSAR.zip -Algorithm SHA256
```

The download is approximately 0.49 GB; extracted CSV files, audit tables and prepared datasets require additional disk space. The audit scans the full release even when preparing a small cohort. It configures DuckDB for four threads and a 4 GB database memory limit; Python and other allocations are additional.

## Fixed preparation contract

### Release cohort (10,000 users)

The V2/V2.1 release uses 10,000 users rather than the historical 2,000-user pilot.
The adapter supports explicit nested cohort sizes. To rebuild this cohort:

```powershell
.\.venv\Scripts\python.exe -m src.data.audit --sizes 10000 --output data/canonical/release
.\.venv\Scripts\python.exe -m src.features.prepare --canonical data/canonical/release/10000 --output data/prepared/release-10000-v1
```

The portable release's optional `trace encode` command compares semantic
vocabulary/feature/cohort/frequency identities in `encoder-contract.json`.
Parquet file bytes may differ across dependency versions; matching encoder
semantics is distinct from claiming identical historical experiment bytes.
Raw inputs use anonymized numeric dataset tokens, not natural-language queries.

- **Cohorts:** defaults are `smoke` with 200 users and `pilot` with 2,000 users; `--sizes` selects other explicit sizes such as the release's 10,000 users. Eligibility requires both services before the training cutoff. Users are ordered by SHA-256 of UTF-8 `42:<raw_user_id>`; smaller cohorts are prefixes of larger ones. The selection seed and eligibility rule are fixed in source.
- **Splits:** timestamps are interpreted in UTC+08:00. Training precedes 29 May 2023 00:00; validation runs from that boundary to 30 May 00:00; test runs from 30 May to 31 May 00:00. Events at or after the final boundary are excluded from scored requests. These are dataset time boundaries.
- **Candidates:** inconsistent search sessions are quarantined. Search keeps `VIDEO`/`IMAGE_ATLAS` candidates and merges repeated items using the maximum click label. Recommendation retains recorded exposures, including duplicates. Missing item metadata does not remove candidates.
- **Features:** histories contain only strictly earlier exposure blocks and issued queries, capped at 50 blocks. Simultaneous events are pooled before being appended to history. Historical feedback, historical search result IDs and user attributes are excluded. Item/token vocabularies are fitted on training data; padding is ID 0 and unknown values use ID 1. The token cap is 50.
- **Catalog assumption:** current candidate captions come from the released static metadata snapshot. Models using these features need an ID-only sensitivity comparison.

`--help` lists the supported path overrides. Keep a new output directory when changing the implementation or policy so previous manifests remain intact.

## Outputs and reproducibility

`raw/` holds the archive and extracted CSVs. Each `canonical/<cohort>/` holds normalized Parquet files, selected users and a checksummed manifest. Each `prepared/<cohort>-v1/` holds requests, history blocks, item metadata, the training vocabulary, feature policy and a checksummed manifest. Audit measurements and dated reports remain local.

The pipeline reads its preparation policy from source code; it does not require the private `.local/configs/study.yaml`. Use the matching training vocabulary when encoding inputs for an exported neural model. Dependency versions can change Parquet serialization and therefore file hashes; a newly generated manifest alone does not prove compatibility with a historical model. The final model bundle includes its selected settings and observed training environment. Exact historical training/evaluation reproduction additionally requires matching original prepared bytes and library behavior; semantic encoder compatibility alone does not establish identical experiment results.

## Attribution and use

Dataset: **KuaiSAR: A Unified Search And Recommendation Dataset**, Zhongxiang Sun and colleagues, 2023 ([paper](https://doi.org/10.1145/3583780.3615123), [release DOI](https://doi.org/10.5281/zenodo.8181109)). TRACE's filtering, cohort selection and temporal preparation modify the released data locally.

The [official project license notice](https://kuaisar.github.io/#license) specifies [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/): give attribution, identify modifications, use the material for noncommercial purposes, and follow ShareAlike terms when distributing adaptations. Public availability does not grant unrestricted commercial use. TRACE references the original download rather than redistributing dataset copies.
