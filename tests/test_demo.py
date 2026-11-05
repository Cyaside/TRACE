"""Demo uses real tensor inference and never sends feedback to the scorer."""
import copy

import pytest
import torch

from src.common import digest
from src.demo import DemoEngine,render,linear_scores
from src.training.trace_v2 import export_weights
from test_trace_v2 import FixtureDataset,config,example,model


def test_demo_export_live_inference_feedback_invariance_and_errors(tmp_path):
    path=tmp_path/"weights.pt"
    c=config("demo")
    net=model(dropout=.1,identity_dropout=.1)
    export_weights(net,FixtureDataset(),c,path)
    baseline={"columns":[0],"mean":[0.],"scale":[1.],"coef":[[1.]],"intercept":[0.]}
    case={"name":"typical","request":example(),"baseline_features":[[0.],[1.],[2.]]}
    bundle={"data_hash":"fixture-v2","model_sha256":digest(path),"baseline_model":baseline,"cases":[case]}
    # Use the actual fixture dataset identity rather than an invented one.
    bundle["data_hash"]=FixtureDataset().data_hash
    engine=DemoEngine(path,bundle,threads=1)
    a=engine.predict(case)
    mutated=copy.deepcopy(case);mutated["request"]["labels"]=[0,1,1]
    b=engine.predict(mutated)
    assert a["trace_logits"]==b["trace_logits"]
    assert a["feedback_used_for_inference"] is False
    assert "Historical feedback is hidden" in render(case,a)
    assert "HISTORICAL EVALUATION after inference" in render(case,a,feedback=True)
    bad=copy.deepcopy(bundle);bad["model_sha256"]="wrong"
    with pytest.raises(ValueError,match="checksum"):DemoEngine(path,bad)
    bad=copy.deepcopy(case);bad["request"]["history"][0]["timestamp_ms"]=bad["request"]["timestamp_ms"]
    with pytest.raises(ValueError,match="earlier"):engine.predict(bad)
