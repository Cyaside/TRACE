"""One-time held-out evaluation after validation-only candidate freezing."""
from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

from src.common import ROOT,digest,read_json,write_json
from src.demo import linear_scores
from src.evaluation.evaluate import evaluate_predictions
from src.evaluation.v2_study import extractor_for,feature_rows,jsonl
from src.features.v2_dataset import V2Dataset
from src.serving.runtime import TraceRanker
from src.serving.encoder import encoder_contract


def evaluate_frozen(data,weights,frozen_path,baseline_model,output,threads=2):
    source_files=("src/models/trace_v2.py","src/serving/runtime.py","src/evaluation/final_test.py","src/evaluation/evaluate.py")
    implementation={name:digest(ROOT/name) for name in source_files}
    frozen=read_json(Path(frozen_path))
    if frozen.get("format")!="trace-final-candidate-v1" or frozen.get("test_used_for_selection") is not False:
        raise ValueError("Freeze a validation-only final candidate before opening test outcomes")
    if digest(Path(weights))!=frozen["weights_sha256"] or digest(Path(baseline_model))!=frozen["baseline_sha256"]:
        raise ValueError("Frozen model/baseline bytes changed")
    output=Path(output)
    report_path=output/"final-test.json"
    if report_path.exists():
        report=read_json(report_path)
        if report["model_sha256"]!=frozen["weights_sha256"] or report["data_hash"]!=frozen["data_hash"]:
            raise ValueError("A different final test was already recorded")
        if report["implementation_hashes"]!=implementation:
            raise ValueError("Final evaluation implementation changed; review the frozen protocol")
        for name,expected in report["prediction_hashes"].items():
            if digest(output/name)!=expected:raise ValueError("Final prediction evidence changed")
        return report
    output.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    dataset=V2Dataset(data)
    if dataset.data_hash!=frozen["data_hash"]:raise ValueError("Final evaluation prepared-data mismatch")
    input_contract=encoder_contract(dataset)
    baseline=read_json(Path(baseline_model))
    ranker=TraceRanker(weights,expected_sha256=frozen["weights_sha256"],expected_data_hash=dataset.data_hash,threads=threads)
    # Vocabulary/frequency came from the original training-only reader above.
    # Only now, after freezing, are the held-out outcome rows opened.
    dataset.rows=pq.read_table(Path(data)/"requests.parquet",filters=[("split","==","test")])
    tasks=dataset.rows["task"].to_numpy()
    extractor=extractor_for(dataset)
    counts={}
    paths={name:output/f"{name}-test.jsonl" for name in ("trace","baseline")}
    handles={name:path.open("w",encoding="utf-8") for name,path in paths.items()}
    import json
    try:
        for task in ("search","recommendation"):
            indexes=np.flatnonzero(tasks==task)
            counts[task]=len(indexes)
            for number,index in enumerate(indexes,1):
                row=dataset.encode(int(index))
                prediction=ranker.rank(row)
                logits={"trace":prediction["scores_in_input_order"],
                        "baseline":linear_scores(feature_rows(extractor,row,dataset),baseline)}
                common={key:row[key] for key in ("task","user_raw","request_id","cohort")}
                for name,scores in logits.items():
                    records=([{**common,"item_raw":row["item_raw"],"labels":row["labels"],"scores":scores}]
                             if task=="search" else
                             [{**common,"item_raw":item,"label":label,"score":score}
                              for item,label,score in zip(row["item_raw"],row["labels"],scores)])
                    for record in records:handles[name].write(json.dumps(record,allow_nan=False)+"\n")
                if number%5000==0:
                    print({"phase":"final_test","task":task,"requests":number,"total_requests":len(indexes)},flush=True)
    finally:
        for handle in handles.values():handle.close()
    metrics=evaluate_predictions(jsonl(paths["trace"]))
    baseline_metrics=evaluate_predictions(jsonl(paths["baseline"]))
    if not metrics["search"]["overall"]["eligible_requests"] or not metrics["recommendation"]["count"]:
        raise ValueError("Final test is empty or has no eligible ranking groups")
    result={"status":"complete","split":"test","model_sha256":frozen["weights_sha256"],
        "data_hash":dataset.data_hash,"baseline_sha256":frozen["baseline_sha256"],
        "metrics":metrics,"baseline_metrics":baseline_metrics,"requests":counts,
        "prediction_hashes":{p.name:digest(p) for p in paths.values()},
        "seconds":time.perf_counter()-started,"model_selection_changed":False,
        "selection_policy":"candidate frozen before test; these results are report-only, never tuning feedback"}
    result["encoder_contract"]=input_contract
    result["implementation_hashes"]=implementation
    write_json(report_path,result)
    return result
