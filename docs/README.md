# Project organization

The main tree contains implementation, reusable commands and synthetic tests. The local workspace contains draft plans and execution evidence.

- `src/data`, `src/features`: release adapters and time-correct request histories.
- `src/models`, `src/training`: rankers, optimization and checkpoint handling.
- `src/evaluation`: shared candidate-ranking and exposure metrics.
- `src/serving`: serving code as it is implemented.
- `scripts`: orchestration, read-only verification and local report generation.
- `tests`: synthetic test source; keep its execution logs local.
- `data`: source references and rebuild instructions; downloaded and generated datasets stay local and ignored.
- `artifacts/models`: explicitly approved trained model exports handled by Git LFS.
- `artifacts/runs`: ignored resume bundles and internal execution records.
- `.local/guide`, `.local/configs`: current internal study plans and contract.
- `.local/reports/YYYY-MM-DD`: combined daily evidence, measurements and test records.
- `.local/environment`, `.local/work`, `.local/backups`: runtime snapshots, scratch output and recovery copies.

Only explicitly approved files are prepared for publication. A general request to commit code does not approve releasing internal results or drafts.
