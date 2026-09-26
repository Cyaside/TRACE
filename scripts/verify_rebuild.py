"""Repeat the full pinned-release adapter and prove byte-stable cohort files."""
from src.common import DATA_ROOT,WORK_ROOT,read_json,reports_dir,write_json
from src.data.audit import prepare

if __name__=='__main__':
    names=('smoke','pilot')
    before={n:read_json(DATA_ROOT/'canonical'/n/'manifest.json') for n in names}
    prepare(DATA_ROOT/'raw/KuaiSAR.zip',DATA_ROOT/'canonical')
    after={n:read_json(DATA_ROOT/'canonical'/n/'manifest.json') for n in names}
    if before!=after:
        write_json(WORK_ROOT/'rebuild-mismatch.json',{'before':before,'after':after})
        raise AssertionError('Full raw adapter rebuild changed canonical manifests')
    write_json(reports_dir()/'rebuild-verification.json',{'status':'verified','scope':'complete raw archive adapter rerun, numeric validation, all cohort artifact bytes and hashes',
        'cohorts':{n:{'data_hash':after[n]['data_hash'],'files':after[n]['files'],'counts':after[n]['counts']} for n in names}})
    print('Full raw rebuild: identical cohort manifests and file checksums',flush=True)
