"""Optional raw-ID encoder backed by the user's rebuilt KuaiSAR preparation."""
from __future__ import annotations
import hashlib
import numpy as np


def encoder_contract(dataset):
    keys=("raw_archive_sha256","cohort_hash","vocabulary_hash","feature_policy_hash","item_count","token_count")
    if any(k not in dataset.manifest for k in keys):
        raise ValueError("Prepared manifest lacks the release encoder identities")
    frequency=hashlib.sha256()
    frequency.update(np.asarray(dataset.frequency_ids,dtype="<i8").tobytes())
    frequency.update(np.asarray(dataset.frequencies,dtype="<i8").tobytes())
    return {"format":"trace-encoder-contract-v1",**{k:dataset.manifest[k] for k in keys},
            "frequency_hash":frequency.hexdigest(),"original_prepared_data_hash":dataset.data_hash}


class PreparedEncoder:
    def __init__(self,data,contract):
        from src.features.v2_dataset import V2Dataset
        self.data=V2Dataset(data)
        actual=encoder_contract(self.data)
        if set(contract)!=set(actual):raise ValueError("Incomplete encoder contract")
        # Parquet byte serialization may change across Arrow versions; preserve
        # the original data hash as provenance and compare encoder semantics.
        for name,value in contract.items():
            if name=="original_prepared_data_hash":continue
            if actual.get(name)!=value:raise ValueError(f"Rebuilt encoder identity differs: {name}")

    def encode(self,request):
        data=self.data
        if request.get("task") not in ("search","recommendation"):
            raise ValueError("Raw request task must be search or recommendation")
        if not request.get("item_raw"):raise ValueError("Raw request needs candidates")
        raw_query=request.get("query_tokens",[])[:50] if request["task"]=="search" else []
        query=set(raw_query)
        captions,features=[],[]
        for raw in request["item_raw"]:
            i=int(np.searchsorted(data.meta_ids,raw))
            caption=(data.metadata["caption_tokens"][int(data.meta_order[i])].as_py()[:50]
                     if i<len(data.meta_ids) and data.meta_ids[i]==raw else [])
            captions.append(data.token_ids(caption))
            words=set(caption);overlap=len(query & words)
            pos=int(np.searchsorted(data.frequency_ids,raw))
            freq=int(data.frequencies[pos]) if pos<len(data.frequency_ids) and data.frequency_ids[pos]==raw else 0
            features.append([overlap/max(1,len(query)),overlap/max(1,len(words)),
                             overlap/max(1,len(query|words)),np.log1p(freq)/data.frequency_scale])
        history=[]
        for block in request.get("history",[])[-50:]:
            history.append({"timestamp_ms":block["timestamp_ms"],"service_mask":block["service_mask"],
                "rec_item_ids":data.lookup(data.items,block.get("rec_items",[])),
                "search_query_ids":[data.token_ids(q) for q in block.get("search_queries",[])]})
        return {"task":request["task"],"user_raw":request.get("user_raw","runtime"),
            "request_id":request.get("request_id","runtime"),"timestamp_ms":request["timestamp_ms"],
            "cohort":"ambiguous","item_raw":request["item_raw"],"item_ids":data.lookup(data.items,request["item_raw"]),
            "caption_ids":captions,"query_ids":data.token_ids(raw_query),"relevance_features":features,"history":history}
