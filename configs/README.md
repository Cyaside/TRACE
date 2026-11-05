# Reusable configurations

Portable model/training configurations live here. `trace-v2-kaggle.json` records the original V2 run. `trace-v21-study.json` declares the bounded ranking-loss/control batch, including checkpoint selection and recommendation guardrails. Running a packager does not itself submit or train a job.

CLI arguments configure data preparation and baseline runs; use each command's `--help` for its options. Configuration files do not imply completed experiments or permission to commit/publish them.

The working study contract is stored privately at `.local/configs/study.yaml`. It includes internal dates, decisions and measured provenance.
