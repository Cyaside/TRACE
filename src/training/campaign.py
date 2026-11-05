"""One bounded, matched three-run study. No sweep or automatic resubmission."""
from __future__ import annotations
import copy
import gc
from pathlib import Path
import time

import torch

from src.common import write_json
from src.training.trace_v21 import train


def run_campaign(dataset, config, output, source_hash, job_start=None):
    output=Path(output)
    job_start=time.perf_counter() if job_start is None else job_start
    completed=[]
    status="complete"
    expected_initial=None
    for declaration in config["runs"]:
        if time.perf_counter()-job_start >= config["campaign_soft_seconds"]-3600:
            status="time_budget_reached"
            break
        run=copy.deepcopy(declaration)
        run["job_soft_seconds"]=config["campaign_soft_seconds"]
        report=train(dataset,run,output,source_hash=source_hash,job_start=job_start)
        if expected_initial is None:
            expected_initial=report["initial_parameter_sha256"]
        if report["initial_parameter_sha256"] != expected_initial:
            raise ValueError("Matched runs have different initial model parameters")
        completed.append({"run_id":run["run_id"],"ranking_loss":run["ranking_loss"],
            "utility_weight":run["utility_weight"],**report})
        write_json(output/"campaign-progress.json",{"runs":completed,"source_hash":source_hash,
            "status":"in_progress","test_accessed":False})
        if report["status"] != "complete":
            status="incomplete"
            break
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    if len(completed) != len(config["runs"]): status="incomplete"
    result={"status":status,"runs":completed,"planned_runs":len(config["runs"]),
        "completed_steps":sum(r["completed_steps"] for r in completed),
        "same_initial_parameters":expected_initial is not None and len(completed)==len(config["runs"]),
        "all_runs_have_eligible_checkpoint":len(completed)==len(config["runs"]) and all(r["selection_eligible"] for r in completed),
        "test_accessed":False,"data_hash":dataset.data_hash,"source_hash":source_hash,
        "job_elapsed_seconds":time.perf_counter()-job_start,
        "comparison_allowed":status=="complete" and all(r["completed_steps"]==r["planned_steps"] for r in completed)}
    write_json(output/"campaign-result.json",result)
    return result
