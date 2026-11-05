"""Release freezing, one-time test gate, portable inference and asset integrity."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.common import ROOT,digest,object_hash,read_json,write_json
from src.evaluation.final_test import evaluate_frozen
from src.features.v2_dataset import V2Dataset
from src.models.trace_v2 import TraceV2
from src.release import assemble_release,freeze_candidate,synthetic_demo
from src.serving.runtime import TraceRanker,bundle_path,verify_bundle
from src.serving.encoder import PreparedEncoder,encoder_contract
from src.training.trace_v2 import export_weights
from test_v2_dataset import prepared


def fixture(tmp_path):
    data=tmp_path/"data";data.mkdir()
    prepared(data)
    rows=pq.read_table(data/"requests.parquet").to_pylist()
    rows.append({**rows[3],"split":"test","request_id":"test-rec"})
    pq.write_table(pa.Table.from_pylist(rows),data/"requests.parquet")
    write_json(data/"vocabulary.json",{"items":[20,30,40,50,60,70],"tokens":[100,200,300,400,500,600]})
    manifest=read_json(data/"manifest.json")
    manifest["files"]={name:digest(data/name) for name in manifest["files"]}
    manifest.update(item_count=8,token_count=8,data_hash=object_hash(manifest["files"]))
    manifest.update(raw_archive_sha256="fixture",cohort_hash="fixture",feature_policy_hash="fixture",
                    vocabulary_hash=object_hash(read_json(data/"vocabulary.json")))
    write_json(data/"manifest.json",manifest)
    dataset=V2Dataset(data)
    architecture=dict(embedding_dim=8,hidden_dim=16,heads=2,ff_dim=32,dropout=0.,max_history=50,identity_dropout=0.,residual_bound=2.)
    net=TraceV2(8,8,**architecture)
    weights=tmp_path/"model.pt"
    export_weights(net,dataset,{"model":architecture},weights)
    baseline=tmp_path/"baseline.json"
    write_json(baseline,{"columns":list(range(17)),"mean":[0.]*17,"scale":[1.]*17,
        "coef":[[0.]*17],"intercept":[0.]})
    campaign=tmp_path/"campaign";campaign.mkdir()
    runs=[]
    for i in range(3):
        run_id=f"run{i}";folder=campaign/run_id;folder.mkdir()
        (folder/"model-best.pt").write_bytes(weights.read_bytes())
        metric={"step":4,"selection_eligible":True,"selection_key":[1,.50+i*.01,-.6],
            "recommendation":{"average_precision":.72,"log_loss":.61},
            "search":{"overall":{"user_macro":{"ndcg@10":.50+i*.01}}}}
        write_json(folder/"validation-best.json",metric)
        write_json(folder/"config.json",{"source_hash":"source","data_hash":dataset.data_hash,
            "config":{"recommendation_ap_floor":.70,"recommendation_log_loss_ceiling":.63}})
        runs.append({"run_id":run_id,"completed_steps":4,"planned_steps":4})
    write_json(campaign/"campaign-result.json",{"status":"complete","comparison_allowed":True,
        "same_initial_parameters":True,"planned_runs":3,"runs":runs,"data_hash":dataset.data_hash,"source_hash":"source"})
    return data,weights,baseline,campaign,dataset.data_hash


def test_candidate_is_frozen_before_test_and_cannot_be_switched(tmp_path):
    data,weights,baseline,campaign,data_hash=fixture(tmp_path)
    frozen,chosen=freeze_candidate(campaign,data_hash,tmp_path/"frozen.json",baseline,"source")
    assert frozen["run_id"]=="run2" and not frozen["test_used_for_selection"]
    metric=read_json(campaign/"run0/validation-best.json")
    metric["selection_key"][1]=.99
    write_json(campaign/"run0/validation-best.json",metric)
    with pytest.raises(ValueError,match="already frozen"):
        freeze_candidate(campaign,data_hash,tmp_path/"frozen.json",baseline,"source")
    with pytest.raises(ValueError,match="source differs"):
        freeze_candidate(campaign,data_hash,tmp_path/"other.json",baseline,"wrong")


def test_final_test_is_bound_to_frozen_weights_and_is_reused(tmp_path):
    data,weights,baseline,campaign,data_hash=fixture(tmp_path)
    frozen,chosen=freeze_candidate(campaign,data_hash,tmp_path/"frozen.json",baseline,"source")
    result=evaluate_frozen(data,chosen,tmp_path/"frozen.json",baseline,tmp_path/"evaluation",threads=1)
    assert result["status"]=="complete" and result["split"]=="test"
    assert not result["model_selection_changed"]
    assert result["metrics"]["recommendation"]["count"]==2
    assert result==evaluate_frozen(data,chosen,tmp_path/"frozen.json",baseline,tmp_path/"evaluation",threads=1)
    (tmp_path/"evaluation/trace-test.jsonl").write_text("changed")
    with pytest.raises(ValueError,match="evidence changed"):
        evaluate_frozen(data,chosen,tmp_path/"frozen.json",baseline,tmp_path/"evaluation",threads=1)


def test_raw_encoder_matches_prepared_features_and_rejects_other_vocab(tmp_path):
    data,weights,baseline,campaign,data_hash=fixture(tmp_path)
    dataset=V2Dataset(data)
    contract=encoder_contract(dataset)
    encoder=PreparedEncoder(data,contract)
    index=int(dataset.indexes[("validation","search")][0])
    original=dataset.rows.slice(index,1).to_pylist()[0]
    raw={k:original[k] for k in ("user_raw","request_id","task","timestamp_ms","item_raw","query_tokens")}
    raw["history"]=[dataset.blocks.slice(i,1).to_pylist()[0] for i in original["history_ids"]]
    encoded=encoder.encode(raw)
    expected=dataset.encode(index)
    for name in ("item_ids","caption_ids","query_ids","relevance_features","history"):
        assert encoded[name]==expected[name]
    assert "labels" not in encoded
    changed={**contract,"vocabulary_hash":"wrong"}
    with pytest.raises(ValueError,match="vocabulary_hash"):
        PreparedEncoder(data,changed)
    # A serialization-only data-hash change is allowed after semantic checks.
    PreparedEncoder(data,{**contract,"original_prepared_data_hash":"another-serialization"})


def test_portable_release_cli_and_sdk_without_private_inputs(tmp_path):
    data,weights,baseline,campaign,data_hash=fixture(tmp_path)
    frozen,chosen=freeze_candidate(campaign,data_hash,tmp_path/"frozen.json",baseline,"source")
    evaluation={"status":"complete","split":"test","model_sha256":frozen["weights_sha256"],
        "data_hash":data_hash,"metrics":{},"baseline_metrics":{},
        "encoder_contract":{"format":"trace-encoder-contract-v1"}}
    wheel=tmp_path/"trace_ranking-2.1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel,"w") as archive:archive.writestr("fixture",b"test-wheel")
    destination=tmp_path/"trace-v2.1.0"
    assemble_release(chosen,frozen,evaluation,wheel,destination,"Fixture card","Fixture readme",ROOT)
    manifest=verify_bundle(destination)
    assert manifest["status"]=="ready" and not manifest["contains_dataset_payload"]
    assert (tmp_path/"trace-v2.1.0.zip").exists()
    with zipfile.ZipFile(destination/"source.zip") as archive:
        assert not any(n.startswith(".local/") or n.endswith(".pt") for n in archive.namelist())
    ranker=TraceRanker.from_bundle(destination,threads=1)
    request=read_json(destination/"request.json")
    output=ranker.rank(request)
    changed={**request,"labels":[1]*len(request["item_ids"])}
    assert output["scores_in_input_order"]==ranker.rank(changed)["scores_in_input_order"]
    invalid=copy.deepcopy(request);invalid["item_ids"][0]=999
    with pytest.raises(ValueError,match="vocabulary index"):ranker.rank(invalid)
    invalid=copy.deepcopy(request);invalid["item_raw"][0]=1.2
    with pytest.raises(ValueError,match="integer identifiers"):ranker.rank(invalid)
    command=[sys.executable,"-m","src","demo","--bundle",str(destination)]
    result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,check=True)
    assert "Synthetic input" in result.stdout and "Reference" in result.stdout
    (destination/"request.json").write_text("corrupted")
    with pytest.raises(ValueError,match="checksum"):verify_bundle(destination)
    with pytest.raises(ValueError,match="leaves"):bundle_path(destination,"../private.txt")


def test_assembly_refuses_missing_or_mismatched_final_evaluation(tmp_path):
    data,weights,baseline,campaign,data_hash=fixture(tmp_path)
    frozen,chosen=freeze_candidate(campaign,data_hash,tmp_path/"frozen.json",baseline,"source")
    with pytest.raises(ValueError,match="Final evaluation"):
        assemble_release(chosen,frozen,{"status":"pending"},tmp_path/"none.whl",tmp_path/"release","","",ROOT)
    assert not (tmp_path/"release").exists()
