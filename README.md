# TRACE

An experimental search and recommendation ranking project using the existing KuaiSAR dataset.

## Project layout

| Folder | Purpose |
|---|---|
| `src/` | Data adapters, temporal features, models, training and evaluation |
| `scripts/` | Experiment, verification and reporting commands |
| `tests/` | Synthetic correctness fixtures and regression tests |
| `configs/` | Reusable configuration files approved for the repository |
| `docs/` | General project structure and usage |
| `data/` | Dataset source/rebuild guide; downloaded and generated files stay local |
| `artifacts/models/` | Model exports eligible for Git LFS |
| `artifacts/runs/` | Ignored resume bundles and internal run records |
| `.local/` | Internal guides, dated evidence, test output, environment snapshots and scratch work |

## Setup

Use Python 3.11 with a compatible PyTorch build:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,train]"
.\.venv\Scripts\python.exe -m pytest -q
```

Data commands use `data/`; training commands use `artifacts/runs/`. Public datasets and their generated derivatives are ignored and rebuilt using the [dataset guide](data/README.md). Exported trained models use `artifacts/models/` and Git LFS. Generated audit reports and experiment ledgers go to `.local/reports/YYYY-MM-DD/`, using the Asia/Jakarta calendar date. Draft study guides are in `.local/guide/` and the current study contract is in `.local/configs/`.

Download the [TRACE 2.1.0 release bundle](https://github.com/Cyaside/TRACE/releases/tag/v2.1.0) for the selected weights, CPU demo and SDK. Its weights are distributed as release assets and do not require Git LFS. Historical V1 exports remain in LFS; to clone source without fetching them, set `GIT_LFS_SKIP_SMUDGE=1` for the clone, or use the bundle's `source.zip`. Dataset preparation is optional for the included demo and downloads the public data once when needed.

For available options, run a command with `--help`, for example:

```powershell
.\.venv\Scripts\python.exe -m src.features.prepare --help
.\.venv\Scripts\python.exe -m scripts.run_baselines --help
```

## Use the trained model

For the installed CLI/SDK and the complete portable model bundle, see
[release usage](docs/release.md). Its commands are `trace demo`, `trace rank`,
`trace encode`, `trace verify` and `trace about`; Python uses
`from trace_ranking import TraceRanker`.

The complete model bundle contains inference weights, a wheel, CPU launcher,
self-contained synthetic demo inputs, model card, final metrics and checksums.
After extracting that bundle:

```powershell
.\run-demo.ps1
.\run-demo.ps1 -Scenario recommendation
```

The first run installs CPU dependencies; later inference runs locally. The demo
uses explicitly synthetic inputs and actual trained weights. Measured quality
is reported separately in the model card and matched final-test metrics.

```python
from trace_ranking import TraceRanker
ranker = TraceRanker.from_bundle("PATH_TO_MODEL_BUNDLE")
result = ranker.rank(encoded_request)
```

Installed commands include `trace verify`, `trace demo`, `trace rank`,
`trace encode` and `trace about`. Dataset-backed real-session replay is an
optional research workflow in [the CLI walkthrough](docs/cli-demo.md); the
portable demo does not depend on private working folders or retraining.

TRACE ranks supplied candidates on anonymized KuaiSAR interactions. Full-catalog
retrieval, natural-language travel/shopping semantics, online CTR uplift and
proven algorithmic novelty are not claims made by this artifact.

## Working rules

Staging, commits and pushes require an explicit user instruction. Internal material under `.local/` stays excluded from Git until selected material is explicitly approved for release. Synthetic test source is separate from private test execution records.

## Attribution

Data: [KuaiSAR](https://kuaisar.github.io/). The neural encoder is an independently implemented [SASRec](https://arxiv.org/abs/1808.09781)-derived adaptation to chronological exposure blocks. Research specifications and experimental evidence are maintained locally.
