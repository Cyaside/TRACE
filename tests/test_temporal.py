import copy
import pytest
from src.features.temporal import attach_prefixes,block_summary,split_at,gap_bucket,cohort_for
from src.data.audit import CUTOFFS,tokens

def fixture():
    blocks=[{"user_raw":1,"timestamp_ms":100,**block_summary([9,7,9],[])},
        {"user_raw":1,"timestamp_ms":200,**block_summary([5],[[3,4],[2]])},
        {"user_raw":1,"timestamp_ms":300,**block_summary([],[[6]])}]
    requests=[{"user_raw":1,"timestamp_ms":200,"task":"search","labels":[1,0]},
        {"user_raw":1,"timestamp_ms":200,"task":"recommendation","labels":[0]},
        {"user_raw":1,"timestamp_ms":300,"task":"search","labels":[1]}]
    return blocks,requests

def test_future_append_preserves_earlier_prefix():
    b,r=fixture();before=attach_prefixes(b,r)
    b.append({"user_raw":1,"timestamp_ms":400,**block_summary([99],[])})
    r.append({"user_raw":1,"timestamp_ms":400,"task":"recommendation"})
    assert attach_prefixes(b,r)[:len(before)]==before

def test_tied_permutation_and_duplicates():
    assert block_summary([1,2,1],[[5],[6]])==block_summary([2,1],[[6],[5]])

def test_score_whole_current_block_before_update():
    b,r=fixture();a=attach_prefixes(b,r)
    assert a[0]["history_ids"]==a[1]["history_ids"]==[0]
    assert a[0]["cohort"]=="transition"
    assert a[2]["cohort"]=="ambiguous" # immediately previous mixed block

def test_current_labels_do_not_change_inputs():
    b,r=fixture();a=attach_prefixes(b,r);changed=copy.deepcopy(r)
    changed[0]["labels"]=[0,1];c=attach_prefixes(b,changed)
    assert {k:v for k,v in a[0].items() if k!="labels"}=={k:v for k,v in c[0].items() if k!="labels"}

def test_exact_boundaries():
    assert [split_at(t) for t in CUTOFFS]==["validation","test","unused"]
    assert split_at(CUTOFFS[0]-1)=="train"

def test_gap_contract():
    assert gap_bucket(None)==0
    assert gap_bucket(60000)==1
    assert gap_bucket(60001)==2
    with pytest.raises(ValueError):gap_bucket(0)

def test_constrained_tokens():
    assert tokens('[]')==[]
    assert tokens('[1, 2]')==[1,2]
    for invalid in ("[1.0]","[-1]","[True]","__import__('os')", "[9223372036854775808]"):
        with pytest.raises(ValueError):tokens(invalid)
