"""Installed TRACE commands for a finalized portable release bundle."""
from __future__ import annotations
import argparse
import os
from pathlib import Path

from src.common import read_json,write_json
from src.serving.runtime import TraceRanker,bundle_path,verify_bundle

VERSION="2.1.0"


def main():
    p=argparse.ArgumentParser(prog="trace",description=__doc__)
    p.add_argument("--version",action="version",version=f"TRACE {VERSION}")
    commands=p.add_subparsers(dest="command",required=True)
    for name in ("verify","demo","rank","about","encode"):
        sub=commands.add_parser(name)
        sub.add_argument("--bundle",type=Path,default=Path(os.environ.get("TRACE_BUNDLE",".")))
        if name in ("demo","rank"):sub.add_argument("--threads",type=int,default=2)
        if name=="demo":sub.add_argument("--scenario",choices=("search","recommendation"),default="search")
        if name=="rank":
            sub.add_argument("--input",type=Path,required=True)
            sub.add_argument("--output",type=Path)
        if name=="encode":
            sub.add_argument("--prepared",type=Path,required=True)
            sub.add_argument("--input",type=Path,required=True)
            sub.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    try:
        manifest=verify_bundle(a.bundle)
        if a.command=="verify":
            print(f'TRACE {manifest["version"]}: {len(manifest["files"])} files verified.')
        elif a.command=="about":
            print(bundle_path(a.bundle,"MODEL_CARD.md").read_text(encoding="utf-8"))
        elif a.command=="encode":
            from src.serving.encoder import PreparedEncoder
            contract=read_json(bundle_path(a.bundle,"encoder-contract.json"))
            encoder=PreparedEncoder(a.prepared,contract)
            if a.output.resolve() in {bundle_path(a.bundle,n) for n in manifest["files"]} | {(a.bundle/"release.json").resolve()}:
                raise ValueError("Output would overwrite a release asset")
            write_json(a.output,encoder.encode(read_json(a.input)))
            print(f"Encoded raw-ID request saved: {a.output}")
        else:
            ranker=TraceRanker.from_bundle(a.bundle,a.threads)
            if a.command=="rank":
                value=read_json(a.input)
                request=value.get("request",value)
                result=ranker.rank(request)
                if a.output:
                    if a.output.resolve() in {bundle_path(a.bundle,n) for n in manifest["files"]} | {(a.bundle/"release.json").resolve()}:
                        raise ValueError("Output would overwrite a release asset")
                    write_json(a.output,result)
                else:
                    import json
                    print(json.dumps(result,indent=2,allow_nan=False))
            else:
                bundle=read_json(bundle_path(a.bundle,"demo.json"))
                case=next(c for c in bundle["cases"] if c["name"]==a.scenario)
                result=ranker.rank(case["request"])
                print(f'TRACE {manifest["version"]} | {a.scenario} | CPU live inference')
                print("Synthetic input example; no benchmark outcome or real-user improvement is implied.")
                print(f'History blocks: {len(case["request"]["history"])} | candidates: {len(result["ranked_items"])}')
                print(f'Query token indices: {case["request"]["query_ids"]}')
                print(f'Cross-history gate: {result["gate"]:.3f}')
                reference=case["request"]["relevance_features"]
                reference_scores={str(item):f[2]+f[3] for item,f in zip(case["request"]["item_raw"],reference)}
                reference_order=sorted(reference_scores,key=lambda item:(-reference_scores[item],int(item)))
                reference_ranks={item:i+1 for i,item in enumerate(reference_order)}
                print("Reference ranking: simple overlap + frequency feature rule on these synthetic inputs.")
                print(f'{"TRACE":>5} {"Reference":>10} {"Item ID":>12} {"Score":>10}')
                for row in result["ranked_items"][:10]:
                    print(f'{row["rank"]:>5} {reference_ranks[str(row["item_raw"])]:>10} {str(row["item_raw"]):>12} {row["score"]:>10.5f}')
                print(f'Inference pipeline: {result["pipeline_seconds"]*1000:.2f} ms; model load: {ranker.load_seconds:.2f} s')
    except (ValueError,FileNotFoundError,KeyError,TypeError,StopIteration) as error:
        p.exit(2,f"TRACE: {error}\n")


if __name__=="__main__":main()
