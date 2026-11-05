# TRACE CLI walkthrough

For the portable installed demo and SDK, start with [release usage](release.md).
That bundle includes synthetic encoded inputs and needs neither training nor a
dataset download. The optional workflow below replays real benchmark sessions
from locally rebuilt data; those session files remain private.

The session replay loads trained TRACE inference weights and ranks complete
candidate lists from prepared snapshots. KuaiSAR item IDs and query tokens are
anonymized; the replay displays those IDs.

## Run an existing local bundle

From the project root in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m src.demo --list
.\.venv\Scripts\python.exe -m src.demo --case helpful
.\.venv\Scripts\python.exe -m src.demo --case helpful --show-feedback
.\.venv\Scripts\python.exe -m src.demo --case harmful --show-feedback
.\.venv\Scripts\python.exe -m src.demo --case typical --show-feedback
```

The default weights path is
`artifacts/models/trace-v2-10k-seed42/model-best.pt`. The default session bundle is
`.local/demo/trace-v2/examples.json`. Use `--weights` and `--examples` to change
them. Model checksum and prepared-data identity must match the bundle.

The table shows the union of both top-10 lists. A positive `Move` means the item
moved upward under TRACE. The comparator is the trained logistic metadata
baseline, scored from the same prepared candidate features at runtime.

Historical clicks are hidden until `--show-feedback` is requested. The scorer
does not receive those labels. It produces uncalibrated logits, not calibrated
click probabilities. The gate describes a model computation, not a causal user
explanation. Nothing is retrained when running the command.

## Build a session bundle

Prepare the dataset using [the data guide](../data/README.md), then supply the
saved validation predictions from TRACE and the matched logistic baseline:

```powershell
.\.venv\Scripts\python.exe -m scripts.build_demo_examples `
  --data data/prepared/teacher-10000-v1 `
  --trace-predictions PATH_TO_TRACE_VALIDATION_JSONL `
  --baseline-predictions PATH_TO_LOGISTIC_VALIDATION_JSONL `
  --baseline-model PATH_TO_LOGISTIC_MODEL_JSON
```

`helpful` is the largest positive request-NDCG difference, `harmful` the most
negative, and `typical` the smallest absolute difference, with deterministic ID
tie breaking. Eligible cases have at least five candidates and mixed labels.
These are deliberately selected demonstration cases, not a representative model
quality sample. Use complete-cohort reports for performance claims.

The generated bundle includes actual released IDs, encoded histories and
historical outcomes. It stays in ignored `.local/` and is not published by the
build command. Dataset attribution/license still applies. The prepared dataset,
inference weights and saved predictions must describe the same frozen contract.

## CPU measurements

```powershell
.\.venv\Scripts\python.exe -m src.demo --case typical `
  --benchmark-output .local/reports/DATE/demo-cpu-benchmark.json
```

The benchmark reports import/verification/model-load time separately from warm
forward and pipeline p50/p95. The prepared-snapshot pipeline includes collation,
model inference, logistic scoring and sorting, and excludes raw dataset encoding
and terminal rendering. Candidate/history sizes, thread count, library version
and observed RSS accompany the result. This is a local workload measurement;
production latency or throughput requires a representative serving benchmark.

## Problem and evaluation scope

TRACE ranks supplied candidates using an earlier search/recommendation history.
Candidate retrieval from an entire catalog and a bandit exploration policy are
separate capabilities. The benchmark concerns KuaiSAR video interactions; the
weights do not gain shopping/travel semantics from renamed demo labels.

Final release metrics are recorded in the bundle's MODEL_CARD.md and metrics.json
after validation-only model selection and held-out evaluation. Session replay
cases do not select a model or replace that evaluation. Offline ranking metrics
do not establish live CTR, revenue or retention changes. Detailed dated working
records remain private.
