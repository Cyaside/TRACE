"""Build the V2.1 study using verified V2 bootstrap and existing private input."""
from __future__ import annotations
import argparse
import base64
import io
from pathlib import Path
import re
import zipfile

from scripts.package_trace_v2_kaggle import build as build_base, SOURCE_FILES
from src.common import ROOT,digest,object_hash,read_json,write_json

EXTRA_FILES=("src/models/ranking.py","src/training/trace_v21.py","src/training/campaign.py",
             "configs/trace-v21-study.json")


def build(destination,username):
    destination=Path(destination)
    report=build_base(destination,username)
    source_files=(*SOURCE_FILES,*EXTRA_FILES)
    hashes={name:digest(ROOT/name) for name in source_files}
    source_hash=object_hash(hashes)
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,"w",compression=zipfile.ZIP_DEFLATED) as archive:
        for name in source_files:
            info=zipfile.ZipInfo(name,(2026,11,5,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            archive.writestr(info,(ROOT/name).read_bytes())
    job=destination/"job/job.py"
    text=job.read_text(encoding="utf-8")
    for name,value in (("SOURCE_HASH",source_hash),("SOURCE_HASHES",hashes),
                       ("BUNDLE",base64.b64encode(buffer.getvalue()).decode())):
        text=re.sub(rf"^{name} = .*?$",lambda _:f"{name} = {value!r}",text,flags=re.MULTILINE)
    text=text.replace("from src.training.trace_v2 import train","from src.training.campaign import run_campaign")
    text=text.replace("(ROOT / 'configs/trace-v2-kaggle.json')","(ROOT / 'configs/trace-v21-study.json')")
    text=text.replace("report = train(dataset, config, OUTPUT, source_hash=SOURCE_HASH, job_start=START)",
                      "report = run_campaign(dataset, config, OUTPUT, source_hash=SOURCE_HASH, job_start=START)")
    compile(text,str(job),"exec")
    job.write_text(text,encoding="utf-8")
    meta=read_json(destination/"job/kernel-metadata.json")
    meta.update(id=f"{username}/trace-v2-1-ranking-controls",title="TRACE V2.1 Ranking Controls")
    write_json(destination/"job/kernel-metadata.json",meta)
    report.update(source_hash=source_hash,source_files=hashes,bootstrap_sha256=digest(job),
        kernel_id=meta["id"],kernel_url=f'https://www.kaggle.com/code/{meta["id"]}',
        config_hash=object_hash(read_json(ROOT/"configs/trace-v21-study.json")),
        campaign_config_hash=object_hash(read_json(ROOT/"configs/trace-v21-study.json")),
        planned_runs=3,planned_updates_per_run=4000,input_dataset_reused=True,submitted=False)
    write_json(destination/"bundle-manifest.json",report)
    print({k:report[k] for k in ("kernel_id","source_hash","planned_runs","submitted")})
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--destination",type=Path,required=True)
    p.add_argument("--username",required=True)
    a=p.parse_args();build(a.destination,a.username)
