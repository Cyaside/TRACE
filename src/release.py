"""Frozen candidate selection and integrity-checked portable release assembly."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import zipfile
import importlib.metadata
import platform
import sys

from src.common import digest,object_hash,read_json,write_json
from src.serving.runtime import bundle_path,verify_bundle


def freeze_candidate(campaign_root, expected_data_hash, output, baseline_model, expected_source_hash=None):
    """Select using validation only. Calling again cannot change a frozen model."""
    root=Path(campaign_root)
    campaign=read_json(root/"campaign-result.json")
    if campaign["status"]!="complete" or not campaign["comparison_allowed"]:
        raise ValueError("Final selection requires the declared campaign to finish")
    if campaign["data_hash"]!=expected_data_hash or not campaign["same_initial_parameters"]:
        raise ValueError("Campaign data/initial-weight identity is invalid")
    if expected_source_hash and campaign["source_hash"]!=expected_source_hash:
        raise ValueError("Campaign source differs from the submitted tested bundle")
    if len(campaign["runs"])!=campaign["planned_runs"]:
        raise ValueError("Declared runs are missing")
    choices=[]
    for number,run in enumerate(campaign["runs"]):
        if run["completed_steps"]!=run["planned_steps"]:
            raise ValueError("A declared training run is incomplete")
        metrics=read_json(root/run["run_id"]/"validation-best.json")
        if metrics.get("selection_eligible") is not True:continue
        config=read_json(root/run["run_id"]/"config.json")
        if config["source_hash"]!=campaign["source_hash"] or config["data_hash"]!=expected_data_hash:
            raise ValueError("Run provenance differs from its campaign")
        declared=config["config"]
        rec=metrics["recommendation"]
        if rec["average_precision"]<declared["recommendation_ap_floor"] or rec["log_loss"]>declared["recommendation_log_loss_ceiling"]:
            raise ValueError("Candidate does not pass its declared guardrails")
        key=metrics["selection_key"]
        if key[0]!=1:raise ValueError("Inconsistent eligible selection key")
        choices.append((key,-number,run,metrics))
    if not choices:
        raise ValueError("No candidate passed the frozen recommendation guardrails; do not promote a fallback")
    _,_,run,metrics=max(choices,key=lambda x:(x[0],x[1]))
    weights=root/run["run_id"]/"model-best.pt"
    config=root/run["run_id"]/"config.json"
    selected_config=read_json(config)
    environment={k:selected_config[k] for k in ("torch","gpu_names","devices") if k in selected_config}
    setup_path=root/"setup.json"
    if setup_path.exists():
        setup=read_json(setup_path)
        if setup["source_hash"]!=campaign["source_hash"]:raise ValueError("Setup provenance differs from campaign")
        environment.update({k:setup[k] for k in ("python","cuda","packages") if k in setup})
    frozen={"format":"trace-final-candidate-v1","run_id":run["run_id"],
        "weights_sha256":digest(weights),"baseline_sha256":digest(Path(baseline_model)),
        "data_hash":expected_data_hash,"source_hash":campaign["source_hash"],
        "validation":metrics,"config_sha256":digest(config),"test_used_for_selection":False,
        "training_config":selected_config["config"],"training_environment":environment,
        "selection":"highest eligible validation user NDCG, BCE tie break, declaration-order tie break"}
    path=Path(output)
    if path.exists() and read_json(path)!=frozen:
        raise ValueError("A different final candidate is already frozen; test outcomes cannot change selection")
    if not path.exists():write_json(path,frozen)
    return frozen,weights


def synthetic_demo():
    """Handwritten inputs, without reading any released dataset or labels."""
    request={"user_raw":"synthetic-user","request_id":"synthetic-search","task":"search",
        "timestamp_ms":2000000,"cohort":"transition","item_raw":[101,102,103,104],
        "item_ids":[2,3,1,4],"query_ids":[2,4],"caption_ids":[[2,4],[3],[],[4]],
        "relevance_features":[[1.,1.,1.,.2],[0.,0.,0.,.8],[0.,0.,0.,0.],[.5,1.,.5,.4]],
        "history":[{"timestamp_ms":1000000,"service_mask":3,"rec_item_ids":[2,3],"search_query_ids":[[2,4]]},
                   {"timestamp_ms":1200000,"service_mask":2,"rec_item_ids":[3,4],"search_query_ids":[]}]}
    rec=copy.deepcopy(request)
    rec.update(task="recommendation",request_id="synthetic-recommendation",query_ids=[])
    rec["relevance_features"]=[[0.,0.,0.,f[3]] for f in rec["relevance_features"]]
    return {"format":"trace-public-demo-v1","synthetic":True,
        "notice":"Invented encoded requests with no labels; live model inference, not an empirical success example.",
        "cases":[{"name":"search","request":request},{"name":"recommendation","request":rec}]}


def assemble_release(weights, frozen, evaluation, wheel, destination, model_card, readme, source_root):
    """Build a ready archive only after frozen-model evaluation has completed.

    Datasets, private session snapshots, optimizer state and dated ledgers are
    intentionally absent from this explicit artifact allowlist.
    """
    weights,wheel,destination=Path(weights),Path(wheel),Path(destination)
    if frozen.get("test_used_for_selection") is not False:
        raise ValueError("Test-informed candidate selection is forbidden")
    if evaluation.get("status")!="complete" or evaluation["model_sha256"]!=frozen["weights_sha256"]:
        raise ValueError("Final evaluation is missing or belongs to another model")
    if evaluation["data_hash"]!=frozen["data_hash"] or evaluation.get("split")!="test":
        raise ValueError("Final evaluation split/data identity mismatch")
    if evaluation.get("encoder_contract",{}).get("format")!="trace-encoder-contract-v1":
        raise ValueError("Final evaluation encoder contract is missing")
    if digest(weights)!=frozen["weights_sha256"]:
        raise ValueError("Frozen model bytes changed")
    if destination.exists():raise FileExistsError("Release directory already exists; preserve its manifest")
    destination.mkdir(parents=True)
    try:
        shutil.copyfile(weights,destination/"model.pt")
        (destination/"dist").mkdir()
        shutil.copyfile(wheel,destination/"dist"/wheel.name)
        write_json(destination/"demo.json",synthetic_demo())
        write_json(destination/"request.json",synthetic_demo()["cases"][0]["request"])
        write_json(destination/"encoder-contract.json",evaluation["encoder_contract"])
        write_json(destination/"TRAINING_CONFIG.json",frozen["training_config"])
        write_json(destination/"TRAINING_ENVIRONMENT.json",frozen["training_environment"])
        packages={name:importlib.metadata.version(name) for name in ("torch","numpy","duckdb","psutil")}
        write_json(destination/"ENVIRONMENT.json",{"python":sys.version,"platform":platform.platform(),"packages":packages,
            "torch_cpu_install_version":packages["torch"].split("+")[0]})
        (destination/"requirements-runtime.txt").write_text("\n".join(f"{name}=={version}" for name,version in packages.items() if name!="torch")+"\n")
        # Only aggregate, curated results. No per-request predictions or IDs.
        write_json(destination/"metrics.json",{"validation":frozen["validation"],"test":evaluation["metrics"],
            "baseline_test":evaluation["baseline_metrics"],"scope":"one selected seed; offline benchmark only"})
        (destination/"MODEL_CARD.md").write_text(model_card,encoding="utf-8")
        (destination/"README.md").write_text(readme,encoding="utf-8")
        (destination/"NOTICE.md").write_text(
            "# Attribution and distribution\n\nTRACE is an independent implementation using PyTorch. "
            "KuaiSAR training data: Zhongxiang Sun and colleagues, CIKM 2023, https://kuaisar.github.io/ . "
            "The dataset license notice is CC BY-NC-SA 4.0. Dataset files and real session snapshots are not included. "
            "The demo inputs are invented. Dependency licenses remain with their respective authors. "
            "No open-source license is assigned to TRACE source/weights by this private packaging command; "
            "rights remain with their owners. Repository visibility is not changed.\n",encoding="utf-8")
        shutil.copyfile(Path(source_root)/"scripts/templates/run-demo.ps1",destination/"run-demo.ps1")
        source=Path(source_root)
        with zipfile.ZipFile(destination/"source.zip","w",compression=zipfile.ZIP_DEFLATED) as archive:
            # Explicit source selection; never walk the project root for assets.
            names=["pyproject.toml","README.md","data/README.md","docs/cli-demo.md","docs/release.md",
                   "docs/README.md","configs/README.md","artifacts/models/README.md"]
            names += [p.relative_to(source).as_posix() for p in (source/"src").rglob("*.py")
                      if "__pycache__" not in p.parts and p.name!="teacher.py"]
            names += [p.relative_to(source).as_posix() for p in (source/"trace_ranking").glob("*.py")]
            names += [p.relative_to(source).as_posix() for p in (source/"tests").glob("test_*.py")
                      if "teacher" not in p.name]
            names += [p.relative_to(source).as_posix() for p in (source/"scripts").glob("*.py")
                      if "teacher" not in p.name]
            names += ["scripts/templates/run-demo.ps1","configs/trace-v2-kaggle.json","configs/trace-v21-study.json"]
            for name in sorted(set(names)):
                if not (source/name).is_file():raise FileNotFoundError(name)
                archive.write(source/name,name)
        files={p.relative_to(destination).as_posix():digest(p) for p in destination.rglob("*") if p.is_file()}
        manifest={"format":"trace-release-v1","version":"2.1.0","status":"ready","weights":"model.pt",
            "data_hash":frozen["data_hash"],"run_id":frozen["run_id"],"source_hash":frozen["source_hash"],
            "final_evaluation_complete":True,"files":files,"contains_dataset_payload":False,
            "publication_performed":False}
        write_json(destination/"release.json",manifest)
        verify_bundle(destination)
        # The outer archive and its checksum are the upload-ready artifacts.
        zip_path=destination.with_name(destination.name+".zip")
        with zipfile.ZipFile(zip_path,"w",compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(destination.rglob("*")):
                if path.is_file():archive.write(path,f"{destination.name}/{path.relative_to(destination).as_posix()}")
        checksum=zip_path.with_name(zip_path.name+".sha256")
        checksum.write_text(f"{digest(zip_path)}  {zip_path.name}\n",encoding="utf-8")
        return manifest
    except BaseException:
        # Preserve failed staging for diagnosis; it is not a ready release.
        (destination/"INCOMPLETE.txt").write_text("Release assembly failed; do not publish this directory.\n")
        raise
