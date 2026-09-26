"""Append-only run evidence; failed and resumed runs remain inspectable."""
from __future__ import annotations
import datetime as dt
import json
import subprocess
from pathlib import Path
from src.common import ROOT,digest,object_hash,reports_dir

def provenance(data_manifest=None):
    files=sorted([*ROOT.glob('src/**/*.py'),*ROOT.glob('configs/*.yaml'),
                  *ROOT.glob('.local/configs/*.yaml'),ROOT/'pyproject.toml'])
    code={p.relative_to(ROOT).as_posix():digest(p) for p in files if p.exists()}
    try:
        revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,stderr=subprocess.DEVNULL,text=True).strip()
    except subprocess.CalledProcessError:
        revision=None
    lock=ROOT/'.local/environment/environment-lock.txt'
    return {"code_commit":revision,"code_content_hash":object_hash(code),"environment_lock":digest(lock) if lock.exists() else None,
        **{k:(data_manifest or {}).get(k) for k in ('raw_archive_sha256','canonical_hash','cohort_hash','split_hash','feature_policy_hash','vocabulary_hash')}}

def append_run(event:dict):
    path=reports_dir(ROOT)/'experiment-ledger.jsonl'
    path.parent.mkdir(parents=True,exist_ok=True)
    record={"recorded_at_utc":dt.datetime.now(dt.timezone.utc).isoformat(),**event}
    with path.open('a',encoding='utf-8') as f:
        f.write(json.dumps(record,sort_keys=True,allow_nan=False)+'\n')

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--record',type=Path,required=True)
    a=p.parse_args();append_run(json.loads(a.record.read_text(encoding='utf-8')))
