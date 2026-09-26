# Trained model exports

These files contain the already trained model state and are eligible for Git LFS. Full training bundles remain in ignored `artifacts/runs/`.

Neural `weights.pt` files contain only:

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
