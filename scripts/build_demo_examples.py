"""Build private, reproducible validation snapshots for the real CLI scorer."""
from __future__ import annotations
import argparse
from pathlib import Path

from src.common import ROOT, digest, read_json, write_json
from src.evaluation.evaluate import search_request_metrics
from src.evaluation.v2_study import extractor_for, feature_rows, jsonl
from src.features.v2_dataset import V2Dataset


def build(data,trace_predictions,baseline_predictions,baseline_model,weights,output):
    baseline={(str(r["user_raw"]),str(r["request_id"])):r for r in jsonl(baseline_predictions) if r["task"]=="search"}
    candidates=[]
    for row in jsonl(trace_predictions):
        if row["task"]!="search" or len(row["item_raw"])<5: continue
        metric=search_request_metrics(row)
        if metric["exclusion"] is not None: continue
        key=(str(row["user_raw"]),str(row["request_id"]))
        base=baseline[key]
        if row["item_raw"]!=base["item_raw"] or row["labels"]!=base["labels"]:
            raise ValueError("Demo comparison candidates/labels do not match")
        delta=metric["ndcg"]-search_request_metrics(base)["ndcg"]
        candidates.append((delta,key))
    if not candidates: raise ValueError("No eligible complete validation examples")
    candidates.sort(key=lambda x:(x[0],x[1]))
    choices={"helpful":candidates[-1],"harmful":candidates[0],
             "typical":min(candidates,key=lambda x:(abs(x[0]),x[1]))}
    dataset=V2Dataset(data)
    extractor=extractor_for(dataset)
    index={(str(r["user_raw"]),str(r["request_id"])):i for i in dataset.indexes[("validation","search")]
           for r in dataset.rows.slice(int(i),1).to_pylist()}
    cases=[]
    for name,(delta,key) in choices.items():
        row=dataset.encode(index[key])
        display=[dataset.blocks.slice(int(i),1).to_pylist()[0] for i in row["history_ids"][-4:]]
        request={k:row[k] for k in ("user_raw","request_id","task","timestamp_ms","cohort","item_raw","item_ids",
            "labels","caption_ids","query_ids","relevance_features","history")}
        rule={"helpful":"largest positive request-NDCG difference vs logistic metadata",
              "harmful":"most negative request-NDCG difference vs logistic metadata",
              "typical":"smallest absolute difference, with deterministic user/request tie break"}[name]
        cases.append({"name":name,"selection_rule":rule,"request":request,
            "raw_query_tokens":row["query_tokens"],"display_history":display,
            "baseline_features":feature_rows(extractor,row,dataset).tolist(),"selection_delta_ndcg":delta})
    bundle={"format":"trace-demo-snapshots-v1","data_hash":dataset.data_hash,"model_sha256":digest(Path(weights)),
        "baseline_model":read_json(baseline_model),"cases":cases,
        "selection_scope":"Validation demonstration selected after evaluation, including best/worst/tied examples; not a representative quality sample.",
        "data_license":"KuaiSAR CC BY-NC-SA 4.0; anonymized IDs/tokens; source https://kuaisar.github.io/",
        "contains_historical_feedback":True,"visibility":"private local examples, not approved for repository publication"}
    write_json(Path(output),bundle)
    print(f"Three private session snapshots saved: {output}")
    return bundle


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,default=ROOT/"data/prepared/teacher-10000-v1")
    p.add_argument("--trace-predictions",type=Path,required=True)
    p.add_argument("--baseline-predictions",type=Path,required=True)
    p.add_argument("--baseline-model",type=Path,required=True)
    p.add_argument("--weights",type=Path,default=ROOT/"artifacts/models/trace-v2-10k-seed42/model-best.pt")
    p.add_argument("--output",type=Path,default=ROOT/".local/demo/trace-v2/examples.json")
    args=p.parse_args()
    build(args.data,args.trace_predictions,args.baseline_predictions,args.baseline_model,args.weights,args.output)
