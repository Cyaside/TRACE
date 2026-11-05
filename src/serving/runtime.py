"""CPU inference on an explicit encoded request; no dataset or training state."""
from __future__ import annotations

import math
from pathlib import Path
import time

from src.common import digest, read_json


def bundle_path(root, name):
    root=Path(root).resolve()
    path=(root/name).resolve()
    if Path(name).is_absolute() or not path.is_relative_to(root):
        raise ValueError("Asset path leaves the release bundle")
    return path


def verify_bundle(root):
    root=Path(root)
    if (root/"INCOMPLETE.txt").exists():raise ValueError("Release assembly is incomplete")
    manifest=read_json(root/"release.json")
    if manifest.get("format")!="trace-release-v1" or manifest.get("status")!="ready":
        raise ValueError("This is not a finalized TRACE release bundle")
    if manifest.get("final_evaluation_complete") is not True:
        raise ValueError("Final model evaluation is incomplete")
    for name,expected in manifest["files"].items():
        path=bundle_path(root,name)
        if not path.is_file() or digest(path)!=expected:
            raise ValueError(f"Release file missing or checksum mismatch: {name}")
    if manifest["weights"] not in manifest["files"]:
        raise ValueError("Weights are absent from the integrity manifest")
    return manifest


class TraceRanker:
    """Scores already encoded model inputs. Original raw IDs need the encoder.

    Item/token IDs are frozen training vocabulary indices, PAD=0/OOV=1. Feature
    encoding belongs to the matching preparation contract; raw natural-language
    queries are not accepted as if they were anonymized benchmark token IDs.
    """
    def __init__(self, weights, *, expected_sha256=None, expected_data_hash=None, threads=2):
        started=time.perf_counter()
        import torch
        from src.models.trace_v2 import TraceV2
        torch.set_num_threads(threads)
        if expected_sha256 and digest(Path(weights))!=expected_sha256:
            raise ValueError("Model checksum mismatch")
        value=torch.load(weights,map_location="cpu",weights_only=True,mmap=True)
        if value.get("format")!="trace-v2-inference-v1":
            raise ValueError("Unsupported inference export")
        if expected_data_hash and value["data_hash"]!=expected_data_hash:
            raise ValueError("Model and encoder data identity differ")
        self.item_count,self.token_count=value["item_count"],value["token_count"]
        self.data_hash=value["data_hash"]
        self.model=TraceV2(self.item_count,self.token_count,**value["architecture"])
        self.model.load_state_dict(value["state_dict"],assign=True)
        self.model.eval()
        self.torch=torch
        self.load_seconds=time.perf_counter()-started

    @classmethod
    def from_bundle(cls,root,threads=2):
        manifest=verify_bundle(root)
        return cls(bundle_path(root,manifest["weights"]),expected_sha256=manifest["files"][manifest["weights"]],
                   expected_data_hash=manifest["data_hash"],threads=threads)

    def rank(self,request):
        from src.models.trace_v2 import collate_v2
        row={k:v for k,v in request.items() if k!="labels"}
        items=row.get("item_ids",[])
        raw=row.get("item_raw",[])
        if not items or len(raw)!=len(items):
            raise ValueError("Nonempty item_ids and matching item_raw are required")
        if row.get("task") not in ("search","recommendation"):
            raise ValueError("Task must be search or recommendation")
        try:
            if any(isinstance(x,bool) or not isinstance(x,(int,str)) for x in raw):
                raise ValueError("Non-integral display ID")
            numeric=[int(x) for x in raw]
        except (ValueError,TypeError,OverflowError):
            raise ValueError("Display item IDs must be finite integer identifiers") from None
        if row["task"]=="search" and len(set(numeric))!=len(numeric):
            raise ValueError("Search candidate IDs must be unique")
        item_values=list(items)
        token_values=list(row.get("query_ids",[]))
        for caption in row.get("caption_ids",[]):token_values.extend(caption)
        for block in row.get("history",[]):
            item_values.extend(block.get("rec_item_ids",[]))
            for query in block.get("search_query_ids",[]):token_values.extend(query)
        for values,maximum,name in ((item_values,self.item_count,"item"),(token_values,self.token_count,"token")):
            if any(isinstance(x,bool) or not isinstance(x,int) or x<0 or x>=maximum for x in values):
                raise ValueError(f"Invalid {name} vocabulary index; use 1 for an unknown identity")
        started=time.perf_counter()
        batch=collate_v2([row],self.model.max_history)
        with self.torch.inference_mode():out=self.model(batch)
        scores=out["final"].tolist()
        if not all(math.isfinite(x) for x in scores):raise ValueError("Non-finite model score")
        order=sorted(range(len(scores)),key=lambda i:(-scores[i],numeric[i],i))
        return {"format":"trace-ranking-output-v1","task":row["task"],"data_hash":self.data_hash,
            "request_id":row.get("request_id"),"gate":float(out["gate"][0]),
            "ranked_items":[{"rank":r+1,"item_raw":raw[i],"score":scores[i]} for r,i in enumerate(order)],
            "scores_in_input_order":scores,"pipeline_seconds":time.perf_counter()-started,
            "feedback_used_for_inference":False}
