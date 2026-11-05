"""Ranking-weight gradients, controlled selection and exact CPU continuation."""
import copy

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from src.models.ranking import delta_ndcg_pairwise
from src.models.trace_v2 import TraceV2,collate_v2
from src.training.trace_v21 import train,task_backward,selection_key
from src.training.campaign import run_campaign
from test_trace_v2 import config,example,FixtureDataset


def cfg(name):
    result=config(name)
    result.update(ranking_loss="delta_ndcg",recommendation_ap_floor=0.,recommendation_log_loss_ceiling=100.,
        selection="maximum_user_macro_search_ndcg_with_fixed_rec_guardrails")
    return result


def test_top_one_weights_move_only_a_pair_affecting_ndcg():
    batch=collate_v2([example()])
    logits=torch.tensor([0.,2.,1.],requires_grad=True)
    loss=delta_ndcg_pairwise(logits,batch,k=1)
    torch.testing.assert_close(loss,F.softplus(torch.tensor(2.)))
    loss.backward()
    s=float(torch.sigmoid(torch.tensor(2.)))
    torch.testing.assert_close(logits.grad,torch.tensor([-s,s,0.]))
    with pytest.raises(ValueError): delta_ndcg_pairwise(logits,batch,cap=0)


def test_weighted_loss_is_microbatch_equivalent_and_degenerate_safe():
    a=example();b=example();b["labels"]=[0,0,0]
    full=collate_v2([a,b])
    logits=torch.tensor([.1,.6,-.3,2.,1.,0.],requires_grad=True)
    loss=delta_ndcg_pairwise(logits,full)
    loss.backward();expected=logits.grad.clone()
    actual=logits.detach().clone().requires_grad_(True)
    split=.5*(delta_ndcg_pairwise(actual[:3],collate_v2([a]))+delta_ndcg_pairwise(actual[3:],collate_v2([b])))
    split.backward()
    torch.testing.assert_close(loss,split)
    torch.testing.assert_close(expected,actual.grad)
    assert torch.isfinite(actual.grad).all()


def test_new_trainer_exact_resume_and_guardrail_failure_flag(tmp_path):
    dataset=FixtureDataset()
    train(dataset,cfg("whole21"),tmp_path)
    train(dataset,cfg("resume21"),tmp_path,stop_after=3)
    train(dataset,cfg("resume21"),tmp_path,resume=tmp_path/"resume21/latest.pt")
    a=torch.load(tmp_path/"whole21/latest.pt",weights_only=False)
    b=torch.load(tmp_path/"resume21/latest.pt",weights_only=False)
    assert a["sampler"]==b["sampler"]
    for name,v in a["models"].items():torch.testing.assert_close(v,b["models"][name],atol=0,rtol=0)
    impossible=cfg("guard-fails")
    impossible["recommendation_ap_floor"]=1.1
    report=train(dataset,impossible,tmp_path)
    assert not report["selection_eligible"]
    assert report["best_selection_key"][0]==0


def test_no_utility_control_does_not_update_gate_from_auxiliary_only():
    c=cfg("aux-control"); c["utility_weight"]=0.
    c["model"].update(dropout=0.,identity_dropout=0.)
    model=TraceV2(8,8,**c["model"])
    stats=task_backward(model,[collate_v2([example()])],"search",c,step=3)
    assert "auxiliary" not in stats
    # The ordinary learned gate still receives direct ranking gradients.
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.gate.parameters())


def test_uniform_reference_matches_original_gradients_and_selection_priorities():
    from src.training.trace_v2 import task_backward as original_backward
    c=cfg("reference");c["ranking_loss"]="uniform"
    c["model"].update(dropout=0.,identity_dropout=0.)
    first=TraceV2(8,8,**c["model"]);second=copy.deepcopy(first)
    batches=[collate_v2([example()])]
    original_backward(first,batches,"search",c,3)
    task_backward(second,batches,"search",c,3)
    for a,b in zip(first.parameters(),second.parameters()):
        if a.grad is not None:torch.testing.assert_close(a.grad,b.grad,atol=0,rtol=0)
    policy={"recommendation_ap_floor":.70,"recommendation_log_loss_ceiling":.63}
    def metric(ndcg,ap=.72,bce=.61):
        return {"recommendation":{"average_precision":ap,"log_loss":bce},"joint_log_loss":bce,
                "search":{"overall":{"user_macro":{"ndcg@10":ndcg}}}}
    passing=selection_key(metric(.51),policy)[0]
    assert passing > selection_key(metric(.99,ap=.69),policy)[0]
    assert selection_key(metric(.52,bce=.62),policy)[0] > passing
    assert not selection_key(metric(.99,bce=.64),policy)[1]


def test_three_run_campaign_shares_initial_weights_and_complete_budgets(tmp_path):
    runs=[]
    for i,(ranking,utility) in enumerate((("uniform",.1),("delta_ndcg",.1),("delta_ndcg",0.))):
        c=cfg(f"campaign{i}");c.update(ranking_loss=ranking,utility_weight=utility)
        runs.append(c)
    result=run_campaign(FixtureDataset(),{"runs":runs,"campaign_soft_seconds":36000},tmp_path,"fixture")
    assert result["status"]=="complete" and result["completed_steps"]==18
    assert result["comparison_allowed"] and result["same_initial_parameters"]
    assert len({r["initial_parameter_sha256"] for r in result["runs"]})==1
