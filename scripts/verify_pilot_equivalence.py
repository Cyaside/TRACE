"""Prove that the draft resource-pilot workload has identical current inputs."""
import pyarrow.parquet as pq
from src.common import DATA_ROOT,read_json,reports_dir,write_json

if __name__=='__main__':
    results={}
    for name in ('smoke','pilot'):
        old=DATA_ROOT/'prepared'/name
        new=DATA_ROOT/'prepared'/(name+'-v1')
        a=read_json(old/'manifest.json');b=read_json(new/'manifest.json')
        checks={n:a['files'][n]==b['files'][n] for n in ('requests.parquet','blocks.parquet','vocabulary.json','feature-policy.json')}
        checks['metadata_values_equal']=pq.read_table(old/'metadata.parquet').sort_by('item_raw').equals(pq.read_table(new/'metadata.parquet').sort_by('item_raw'))
        assert all(checks.values()),checks
        results[name]={'checks':checks,'draft_data_hash':a['data_hash'],'current_data_hash':b['data_hash']}
    write_json(reports_dir()/'pilot-equivalence.json',{'status':'verified','scope':'all request/block/vocabulary/policy bytes identical; all metadata rows equal after key sort; encoding is deterministic and key-based','cohorts':results})
    print('Draft/current workload input equivalence verified',flush=True)
