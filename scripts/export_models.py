"""Export existing trained weights without optimizer state or internal evidence."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import pickle
import re

import torch

from src.common import ARTIFACTS_ROOT, digest, read_json, reports_dir, write_json

ARCHITECTURE_FIELDS = (
    "mode", "embedding_dim", "hidden_dim", "heads", "layers", "ff_dim",
    "dropout", "max_history", "item_count", "token_count",
)


def export_run(run_id: str, destination: Path | None = None) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id):
        raise ValueError("Run ID must be a single safe directory name")
    source = ARTIFACTS_ROOT / "runs" / run_id
    output = (destination or ARTIFACTS_ROOT / "models") / run_id
    output.mkdir(parents=True, exist_ok=True)
    if (source / "best.pt").exists():
        # Inputs are the project's own trusted training snapshots.
        snapshot = torch.load(source / "best.pt", map_location="cpu", weights_only=False)
        config = read_json(source / "config.json")
        if snapshot["config_hash"] != config["config_hash"] or snapshot["data_hash"] != config["data_hash"]:
            raise ValueError("Checkpoint/configuration provenance mismatch")
        payload = {
            "format_version": 1,
            "architecture": {name: config[name] for name in ARCHITECTURE_FIELDS},
            "state_dict": snapshot["models"],
        }
        target = output / "weights.pt"
        temporary = target.with_suffix(".pt.tmp")
        torch.save(payload, temporary)
        os.replace(temporary, target)
        reloaded = torch.load(target, map_location="cpu", weights_only=True)
        assert set(reloaded) == {"format_version", "architecture", "state_dict"}
        assert reloaded["architecture"] == payload["architecture"]
        assert reloaded["state_dict"].keys() == snapshot["models"].keys()
        assert all(torch.equal(tensor, reloaded["state_dict"][name])
                   for name, tensor in snapshot["models"].items())
        evidence = {"source_sha256": digest(source / "best.pt"), "tensor_roundtrip_equal": True}
    elif (source / "model.pkl").exists():
        from src.training.serialization import load_trusted_tabular, serialize_tabular_model
        target = output / "model.pkl"
        model = load_trusted_tabular((source / "model.pkl").read_bytes())
        payload = serialize_tabular_model(model)
        temporary = target.with_suffix(".pkl.tmp")
        temporary.write_bytes(payload)
        os.replace(temporary, target)
        reloaded = pickle.loads(target.read_bytes())
        assert type(reloaded).__module__ == "src.training.tabular"
        assert type(reloaded.extractor).__module__ == "src.training.tabular"
        assert pickle.dumps(reloaded, protocol=pickle.HIGHEST_PROTOCOL) == payload
        original = digest(source / "model.pkl")
        evidence = {"source_sha256": original, "canonical_class_modules": True,
                    "fitted_state_roundtrip_equal": True}
    else:
        raise FileNotFoundError(f"No trained model found for {run_id}")
    return {"run_id": run_id, "export_path": str(target), "export_sha256": digest(target),
            "bytes": target.stat().st_size, **evidence}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", action="append", required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    exports = [export_run(run_id) for run_id in args.run_id]
    write_json(reports_dir() / "exports/model-export.json",
               {"status": "verified", "training_performed": False, "exports": exports})
    print(f"Exported and verified {len(exports)} existing models; no training performed", flush=True)


if __name__ == "__main__":
    main()
