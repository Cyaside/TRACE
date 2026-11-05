# Trained model exports

These files contain the already trained model state and are eligible for Git LFS. Full training bundles remain in ignored `artifacts/runs/`.

## Current TRACE inference exports

The final TRACE 2.1.0 weights are included as `model.pt` in the [portable release ZIP](https://github.com/Cyaside/TRACE/releases/tag/v2.1.0), independently of Git LFS storage. The small `trace-v2.1.0/manifest.json` in this repository records their identity and download location. Model SHA-256: `f054566288cc8f4e53414bb573fce209947e1b3df6884fd2ea145536ba432df8`; size: 198,875,189 bytes. Historical V1 LFS exports remain unchanged.

V2/V2.1 `model-best.pt`/`model-latest.pt` use the `trace-v2-inference-v1` format with architecture settings, item/token vocabulary sizes, `state_dict`, prepared-data identity and feature version. They exclude optimizer, scheduler, sampler/RNG and internal execution ledgers.

The public SDK loads these on CPU with memory mapping:

```python
from trace_ranking import TraceRanker
ranker = TraceRanker("path/to/model-best.pt")
result = ranker.rank(encoded_request)
```

For a finalized portable bundle, use `TraceRanker.from_bundle(path)` to verify its manifest and hashes. Raw-ID input encoding needs the matching training vocabulary/feature contract; the optional `PreparedEncoder` verifies semantic identities after dataset reconstruction. See [release usage](../../docs/release.md).

## Historical V1 exports

V1 neural `weights.pt` files contain:

- `format_version`
- `architecture`: constructor settings and vocabulary sizes
- `state_dict`: learned tensors

They exclude optimizer/scheduler/RNG state, training dates, metrics and internal provenance. Load with `torch.load(path, map_location="cpu", weights_only=True)`, construct `build_models` from the architecture settings, and call `load_state_dict(bundle["state_dict"])`. Use the matching prepared training vocabulary when encoding inputs. The included `v1-*` exports use `data/prepared/pilot-v1/`; rebuild it using the [dataset guide](../../data/README.md). Dataset files are not included in Git.

Tabular `model.pkl` exports retain the fitted model state with canonical `src.training.tabular` class references. Load using ordinary `pickle.load()` from a process that can import the project package, with a compatible scikit-learn runtime. Exporting also converts older CLI snapshots without retraining or changing their fitted state.

To export an existing run without training:

```powershell
.\.venv\Scripts\python.exe -m scripts.export_models --run-id <run-id>
```

Export verification and source provenance are saved privately in the dated local reports folder. Exporting does not stage, commit or upload anything.
