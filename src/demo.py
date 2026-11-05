"""Run TRACE on a small prepared session snapshot, without a web app or training."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

START = time.perf_counter()
from src.common import ROOT, digest, read_json, write_json


def linear_scores(features, model):
    import numpy as np
    columns = model["columns"]
    values = np.asarray(features,dtype=np.float64)[:,columns]
    return (((values-model["mean"])/model["scale"]) @ np.asarray(model["coef"])[0]
            + model["intercept"][0]).tolist()


class DemoEngine:
    def __init__(self, weights, bundle, threads=2):
        tick = time.perf_counter()
        import torch
        from src.models.trace_v2 import TraceV2
        self.torch = torch
        torch.set_num_threads(threads)
        self.bundle = bundle
        if digest(Path(weights)) != bundle["model_sha256"]:
            raise ValueError("Model checksum differs from the session bundle")
        value = torch.load(weights,map_location="cpu",weights_only=True,mmap=True)
        if value["format"] != "trace-v2-inference-v1" or value["data_hash"] != bundle["data_hash"]:
            raise ValueError("Model format/prepared-data identity mismatch")
        self.model = TraceV2(value["item_count"],value["token_count"],**value["architecture"])
        self.model.load_state_dict(value["state_dict"],assign=True)
        self.model.eval()
        self.load_seconds = time.perf_counter()-tick

    def predict(self, case):
        from src.models.trace_v2 import collate_v2
        request = {k:v for k,v in case["request"].items() if k != "labels"}
        # Historical outcomes remain outside the input passed to the scorer.
        tick = time.perf_counter()
        batch = collate_v2([request],self.model.max_history)
        forward_start = time.perf_counter()
        with self.torch.inference_mode():
            values = self.model(batch)
        forward_seconds = time.perf_counter()-forward_start
        trace = values["final"].tolist()
        baseline = linear_scores(case["baseline_features"],self.bundle["baseline_model"])
        if not all(math.isfinite(x) for x in trace+baseline):
            raise ValueError("Non-finite prediction")
        items = request["item_raw"]
        order = lambda scores: sorted(range(len(items)),key=lambda i:(-scores[i],int(items[i])))
        trace_order, baseline_order = order(trace),order(baseline)
        return {"case": case["name"], "request_id": request["request_id"], "task": request["task"],
            "items": items, "trace_logits": trace,"baseline_logits": baseline,
            "trace_order": trace_order,"baseline_order": baseline_order,
            "gate": float(values["gate"][0]),"base_logits": values["base"].tolist(),
            "forward_seconds": forward_seconds,"pipeline_seconds": time.perf_counter()-tick,
            "feedback_used_for_inference": False}


def render(case, result, feedback=False):
    request=case["request"]
    lines=[f'TRACE | {case["name"]} | {request["task"]}',
        f'Anonymous user {request["user_raw"]}; request {request["request_id"]}',
        f'Candidates: {len(result["items"])} | earlier history blocks: {len(request["history"])}',
        f'Raw query tokens (anonymized): {case.get("raw_query_tokens",[])}',"", "RECENT HISTORY (before the request)"]
    for block in case.get("display_history",[])[-4:]:
        lines.append(f'  t={block["timestamp_ms"]} | services={block["service_mask"]} | rec IDs={block["rec_items"][:6]} | issued queries={block["search_queries"][:2]}')
    lines += ["",f'Model inference | cross-history gate: {result["gate"]:.3f}',
        "Scores are uncalibrated ranking logits; a gate is not proof of a causal explanation.",
        "", "SAME CANDIDATES: logistic baseline versus TRACE (union of both top-10 lists)"]
    trace_rank={i:r+1 for r,i in enumerate(result["trace_order"])}
    baseline_rank={i:r+1 for r,i in enumerate(result["baseline_order"])}
    lines.append(f'{"Item ID":>14} {"Baseline":>9} {"TRACE":>7} {"Move":>7}'+("  Historical click" if feedback else ""))
    selected=set(result["trace_order"][:10]) | set(result["baseline_order"][:10])
    for i in sorted(selected,key=lambda i:trace_rank[i]):
        move=baseline_rank[i]-trace_rank[i]
        line=f'{str(result["items"][i]):>14} {baseline_rank[i]:>9} {trace_rank[i]:>7} {move:>+7}'
        if feedback:
            line += f'  {int(request["labels"][i])}'
        lines.append(line)
    if feedback:
        from src.evaluation.evaluate import search_request_metrics
        if request["task"] == "search":
            common={k:request[k] for k in ("user_raw","request_id","item_raw","labels","cohort")}
            a=search_request_metrics({**common,"scores":result["baseline_logits"]})["ndcg"]
            b=search_request_metrics({**common,"scores":result["trace_logits"]})["ndcg"]
            lines += ["",f'HISTORICAL EVALUATION after inference: NDCG@10 baseline={a:.4f}, TRACE={b:.4f}',
                      "This is one selected demonstration case; see full validation results for model quality."]
    else:
        lines += ["", "Historical feedback is hidden. Use --show-feedback to reveal it after inference."]
    return "\n".join(lines)


def benchmark(engine, cases, repeats=30, warmup=5):
    import numpy as np
    import psutil
    if repeats < 1 or warmup < 0:
        raise ValueError("Invalid benchmark counts")
    for i in range(warmup): engine.predict(cases[i%len(cases)])
    observed_rss=psutil.Process().memory_info().rss
    values=[]
    for i in range(repeats):
        values.append(engine.predict(cases[i%len(cases)]))
        observed_rss=max(observed_rss,psutil.Process().memory_info().rss)
    summary={}
    for field in ("forward_seconds","pipeline_seconds"):
        sample=np.asarray([v[field] for v in values])*1000
        summary[field.replace("seconds","milliseconds")]={"p50":float(np.median(sample)),
            "p95":float(np.percentile(sample,95)),"mean":float(sample.mean())}
    return {"warmup_requests":warmup,"measured_requests":repeats,"model_import_verify_load_seconds":engine.load_seconds,
        "warm_latency":summary,"rss_bytes_observed":observed_rss,
        "candidate_counts":sorted(set(len(c["request"]["item_raw"]) for c in cases)),
        "history_block_counts":sorted(set(len(c["request"]["history"]) for c in cases)),
        "torch":str(engine.torch.__version__),"threads":engine.torch.get_num_threads(),"device":"cpu",
        "scope":"Prepared encoded snapshots; includes collation/model/baseline/sorting, excludes raw data encoding and terminal rendering. Not a production SLA.",
        "neural_training":False}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--examples",type=Path,default=ROOT/".local/demo/trace-v2/examples.json")
    p.add_argument("--weights",type=Path,default=ROOT/"artifacts/models/trace-v2-10k-seed42/model-best.pt")
    p.add_argument("--case",choices=("helpful","harmful","typical"),default="typical")
    p.add_argument("--list",action="store_true")
    p.add_argument("--show-feedback",action="store_true")
    p.add_argument("--json-output",type=Path)
    p.add_argument("--benchmark-output",type=Path)
    args=p.parse_args()
    try:
        if not args.examples.exists():
            raise ValueError("Session examples are missing. Build them with python -m scripts.build_demo_examples --help")
        bundle=read_json(args.examples)
        if args.list:
            for case in bundle["cases"]: print(f'{case["name"]}: {case["selection_rule"]}')
            return
        engine=DemoEngine(args.weights,bundle)
        case=next(c for c in bundle["cases"] if c["name"]==args.case)
        result=engine.predict(case)
        print(render(case,result,args.show_feedback))
        if args.json_output: write_json(args.json_output,result)
        if args.benchmark_output:
            report=benchmark(engine,bundle["cases"])
            report["process_elapsed_seconds"]=time.perf_counter()-START
            write_json(args.benchmark_output,report)
            print(f'\nCPU benchmark saved: {args.benchmark_output}')
    except (ValueError,FileNotFoundError,KeyError,StopIteration) as error:
        p.exit(2,f"TRACE demo input error: {error}\n")


if __name__ == "__main__":
    main()
