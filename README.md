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
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest -q
```

Data commands use `data/`; training commands use `artifacts/runs/`. Public datasets and their generated derivatives are ignored and rebuilt using the [dataset guide](data/README.md). Exported trained models use `artifacts/models/` and Git LFS. Generated audit reports and experiment ledgers go to `.local/reports/YYYY-MM-DD/`, using the Asia/Jakarta calendar date. Draft study guides are in `.local/guide/` and the current study contract is in `.local/configs/`.

Install [Git LFS](https://git-lfs.com/) before cloning to retrieve the included trained weights. In an existing checkout, `git lfs pull` retrieves those exports. Dataset preparation downloads the public release once; loading existing model exports does not require retraining.

For available options, run a command with `--help`, for example:

```powershell
.\.venv\Scripts\python.exe -m src.features.prepare --help
.\.venv\Scripts\python.exe -m scripts.run_baselines --help
```

## Working rules

Staging, commits and pushes require an explicit user instruction. Internal material under `.local/` stays excluded from Git until selected material is explicitly approved for release. Synthetic test source is separate from private test execution records.

## Attribution

Data: [KuaiSAR](https://kuaisar.github.io/). The neural encoder is an independently implemented [SASRec](https://arxiv.org/abs/1808.09781)-derived adaptation to chronological exposure blocks. Research specifications and experimental evidence are maintained locally.
