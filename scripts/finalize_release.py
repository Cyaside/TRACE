"""Finish a frozen release after the existing Kaggle study has completed.

Does not submit jobs, tune models, commit, push, or publish to any service.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from src.common import ROOT,digest,read_json,write_json
from src.evaluation.final_test import evaluate_frozen
from src.release import assemble_release,freeze_candidate,synthetic_demo
from src.serving.runtime import TraceRanker,verify_bundle


def build_wheel(output):
    output=Path(output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    (ROOT/".local/work").mkdir(parents=True,exist_ok=True)
    # Build only our own release package from a clean staging tree. Optional
    # historical teacher adapters do not enter the wheel.
    with tempfile.TemporaryDirectory(prefix="trace-wheel-",dir=ROOT/".local/work") as tmp:
        staging=Path(tmp)
        shutil.copyfile(ROOT/"pyproject.toml",staging/"pyproject.toml")
        shutil.copyfile(ROOT/"LICENSE",staging/"LICENSE")
        for source in (ROOT/"src").rglob("*.py"):
            if "__pycache__" in source.parts or source.name=="teacher.py":continue
            destination=staging/source.relative_to(ROOT)
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source,destination)
        for source in (ROOT/"trace_ranking").glob("*.py"):
            destination=staging/source.relative_to(ROOT)
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source,destination)
        subprocess.run([sys.executable,"-m","pip","wheel",".","--no-deps","--no-build-isolation",
                        "--wheel-dir",str(output)],cwd=staging,check=True)
    wheels=list(output.glob("trace_ranking-2.1.0-*.whl"))
    if len(wheels)!=1:raise ValueError("Expected exactly one versioned release wheel")
    return wheels[0]


def card(frozen,evaluation):
    metrics=evaluation["metrics"]
    baseline=evaluation["baseline_metrics"]
    s=metrics["search"]["overall"]
    r=metrics["recommendation"]
    return f'''# TRACE 2.1.0 model card

## Purpose and architecture

Rank supplied search/recommendation candidates using strictly earlier behavior.
The compact Transformer uses isolated same/cross-service streams, a base score,
bounded residual and request gate. Training is direct supervised learning.
This is a CPU-runnable research/portfolio model, not a deployed consumer service.

Selected run: `{frozen['run_id']}`. The candidate was frozen on validation before
the final test was read. Selection used eligible search user-macro NDCG with the
predeclared recommendation guardrails. No test-based model revision was made.
The exact selected settings and observed training platform are included in
TRAINING_CONFIG.json and TRAINING_ENVIRONMENT.json. The code archive contains
the original controlled loss variants and their declared study configuration.

## Final held-out results

| Model | Search user NDCG@10 | Recommendation AP | Rec log loss |
|---|---:|---:|---:|
| TRACE | {s['user_macro']['ndcg@10']:.4f} | {r['average_precision']:.4f} | {r['log_loss']:.4f} |
| Logistic metadata | {baseline['search']['overall']['user_macro']['ndcg@10']:.4f} | {baseline['recommendation']['average_precision']:.4f} | {baseline['recommendation']['log_loss']:.4f} |

Search eligible requests: {s['eligible_requests']}; eligible users:
{s['eligible_users']}. Recommendation exposures: {r['count']}.
These are offline KuaiSAR results from one selected seed. NDCG is ranking
quality, not accuracy; AP is not a live conversion rate. Improvements, novelty
and robustness across training seeds are not implied by releasing the software.

## Data and availability

KuaiSAR-small is anonymized short-video search/recommendation interaction data.
Train precedes 29 May 2023, validation is 29 May and test is 30 May, UTC+8.
Vocabulary/frequency use training only; histories precede the current request.
Static caption metadata has an availability limitation. OOV/missing metadata
candidates are retained. The data is not travel/shopping interaction data.

Dataset source/license: https://kuaisar.github.io/ (CC BY-NC-SA 4.0 notice).
Dataset copies, real session snapshots, optimizer state and private ledgers are
absent from this bundle. Demo requests are invented encoded inputs with no
historical outcomes. They exercise real model inference, not empirical success.

## Interface and limitations

Inputs use the exact encoded feature schema in request.json. Vocabulary index 1
represents an unknown item/token. Natural-language queries or arbitrary raw item
IDs require the matching encoder; renaming an ID does not change its semantics.
Scores are uncalibrated logits. A gate value is not a causal explanation.
Full-catalog retrieval and bandit exploration are outside this artifact.

CPU memory-mapped loading is supported. Latency depends on candidate count,
history length and host load; benchmark descriptions must distinguish encoded
input latency from raw data preparation. This release makes no production SLA.

Weights SHA-256: `{frozen['weights_sha256']}`.
Prepared-data hash: `{frozen['data_hash']}`.
Training source hash: `{frozen['source_hash']}`.
'''


README='''# TRACE 2.1.0

CPU inference with the selected trained ranking model, SDK and CLI. No training
or dataset download is needed for the included synthetic-input demonstration.

## Problem and use

Someone moving between recommendation and search may change intent. TRACE ranks
an existing candidate pool using earlier behavior, query context and a bounded
cross-service contribution. The engineering question is how to use that context
without letting irrelevant history degrade ranking. The output is an ordered
candidate list, not a generated answer or a complete catalog search service.
MODEL_CARD.md explains the method, measured outcomes and limitations; metrics.json
contains the final offline comparison against the trained logistic baseline.

## Windows quick start

Install Python 3.11, extract this archive, then run in PowerShell:

```powershell
.\\run-demo.ps1
.\\run-demo.ps1 -Scenario recommendation
```

The first run creates a local virtual environment and installs CPU PyTorch and
the included wheel; Internet is required for those dependencies. Subsequent
inference is local. CPU setup reference: https://pytorch.org/get-started/locally/ .

## Manual install and commands

```powershell
py -3.11 -m venv .venv
.\\.venv\\Scripts\\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\\.venv\\Scripts\\python.exe -m pip install dist/trace_ranking-2.1.0-py3-none-any.whl
.\\.venv\\Scripts\\trace.exe verify --bundle .
.\\.venv\\Scripts\\trace.exe demo --bundle . --scenario search
.\\.venv\\Scripts\\trace.exe rank --bundle . --input request.json --output result.json
.\\.venv\\Scripts\\trace.exe about --bundle .
```

The demo uses explicitly synthetic encoded inputs and runs real trained weights.
It contains no benchmark labels and makes no claim that the invented scenario
is a measured user outcome. See MODEL_CARD.md and metrics.json for actual held-out
results. Hashes are in release.json; the outer archive has a .sha256 sidecar.

## Python SDK

```python
import json
from trace_ranking import TraceRanker
ranker = TraceRanker.from_bundle(".")
request = json.load(open("request.json", encoding="utf-8"))
result = ranker.rank(request)
print(result["ranked_items"])
```

Encoded item/token IDs are vocabulary indices, not arbitrary raw identifiers.
Input history must be chronological and earlier than timestamp_ms; labels are
never scoring inputs. request.json demonstrates all required fields.

## Reproduce and real-session replay

source.zip includes the project source, training configs, correctness tests and
data reconstruction instructions in data/README.md. Install source with
pip install -e ".[dev,train]" for full development. The real-session replay
builder is documented in docs/cli-demo.md and needs locally prepared KuaiSAR.
No private .local inputs, test predictions or resume bundles are required by
this installed demo. This bundle is prepared privately; publishing and repository
visibility are separate actions. See NOTICE.md for attribution/distribution.

For raw benchmark ID/token inputs, install the optional encoder dependencies:

```powershell
.\\.venv\\Scripts\\python.exe -m pip install "dist/trace_ranking-2.1.0-py3-none-any.whl[train]"
```

The encoder dependencies need the regular package index on a first install. Rebuild
the 10k preparation as described in source.zip/data/README.md, then use
trace encode --bundle . --prepared PATH --input raw-request.json --output encoded.json.
The encoder checks vocabulary, feature-policy, cohort and training-frequency
identities. Different Parquet byte serialization alone does not change those
semantic checks. Inputs remain anonymized benchmark token IDs, not prose.
'''


def check_installed_wheel(wheel,weights,request,output,runtime_python=sys.executable):
    wheel,weights,request,output=map(lambda p:Path(p).resolve(),(wheel,weights,request,output))
    runtime_python=Path(runtime_python).resolve()
    with tempfile.TemporaryDirectory(prefix="trace-installed-",dir=ROOT/".local/work") as tmp:
        subprocess.run([str(runtime_python),"-I","-m","pip","install","--no-deps","--target",tmp,str(wheel)],check=True)
        script='''import json,pathlib,sys
sys.path.insert(0,sys.argv[1])
import src
assert pathlib.Path(src.__file__).resolve().is_relative_to(pathlib.Path(sys.argv[1]).resolve())
from trace_ranking import TraceRanker
ranker=TraceRanker(sys.argv[2])
request=json.loads(pathlib.Path(sys.argv[3]).read_text())
result=ranker.rank(request)
assert result["feedback_used_for_inference"] is False
assert "pyarrow" not in sys.modules and "sklearn" not in sys.modules
pathlib.Path(sys.argv[4]).write_text(json.dumps({"status":"passed","import_path":src.__file__,"scores":result["scores_in_input_order"],"optional_training_imports":False},indent=2)+"\\n")
'''
        subprocess.run([str(runtime_python),"-I","-c",script,tmp,str(weights),str(request),str(output)],cwd=Path(tmp),check=True)
    return read_json(output)


def finalize(campaign,submitted,data,baseline_model,output,work,runtime_python=sys.executable):
    campaign,submitted,data,baseline_model,output,work=map(lambda p:Path(p).resolve(),
        (campaign,submitted,data,baseline_model,output,work))
    runtime_python=Path(runtime_python).resolve()
    submitted=read_json(submitted)
    work=Path(work);work.mkdir(parents=True,exist_ok=True)
    frozen,weights=freeze_candidate(campaign,submitted["data_hash"],work/"frozen-candidate.json",
                                    baseline_model,submitted["source_hash"])
    evaluation=evaluate_frozen(data,weights,work/"frozen-candidate.json",baseline_model,work/"evaluation")
    wheel=build_wheel(work/"dist")
    write_json(work/"synthetic-input.json",synthetic_demo()["cases"][0]["request"])
    smoke=check_installed_wheel(wheel,weights,work/"synthetic-input.json",work/"installed-wheel-test.json",runtime_python)
    import numpy as np
    expected=TraceRanker(weights).rank(read_json(work/"synthetic-input.json"))
    np.testing.assert_allclose(smoke["scores"],expected["scores_in_input_order"],atol=2e-4,rtol=2e-4)
    manifest=assemble_release(weights,frozen,evaluation,wheel,output,card(frozen,evaluation),README,ROOT)
    ranker=TraceRanker.from_bundle(output)
    ranker.rank(read_json(Path(output)/"request.json"))
    verify_bundle(output)
    write_json(work/"release-completion.json",{"status":"ready","archive":str(Path(output).with_name(Path(output).name+".zip")),
        "manifest":manifest,"publication_performed":False})
    return manifest


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--campaign",type=Path,required=True)
    p.add_argument("--submitted-manifest",type=Path,required=True)
    p.add_argument("--data",type=Path,required=True)
    p.add_argument("--baseline-model",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--work",type=Path,required=True)
    p.add_argument("--runtime-python",type=Path,default=Path(sys.executable))
    a=p.parse_args()
    finalize(a.campaign,a.submitted_manifest,a.data,a.baseline_model,a.output,a.work,a.runtime_python)
