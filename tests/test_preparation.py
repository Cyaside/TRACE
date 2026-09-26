from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from src.common import digest,write_json
from src.data.audit import CUTOFFS
from src.features.prepare import build
from src.features.dataset import PreparedDataset

def test_train_only_vocabulary_and_reproducible_prep(tmp_path,monkeypatch):
    import src.features.prepare as preparation
    monkeypatch.setattr(preparation,'ROOT',tmp_path)
    canonical=tmp_path/'canonical';canonical.mkdir()
    t=CUTOFFS[0]
    search=[{'user_raw':1,'session_raw':1,'timestamp_ms':t-1000,'query_signature':'[3]', 'item_raw':[7,8], 'labels':[1,0],'raw_count':2,'retained_raw_count':2},
        {'user_raw':1,'session_raw':2,'timestamp_ms':t+1000,'query_signature':'[99]','item_raw':[9,7],'labels':[0,1],'raw_count':2,'retained_raw_count':2}]
    rec=[{'user_raw':1,'item_raw':7,'timestamp_ms':t-2000,'label':1,'raw_row':1},
        {'user_raw':1,'item_raw':8,'timestamp_ms':t-2000,'label':0,'raw_row':2}]
    metadata=[{'item_raw':7,'caption':'[11, 12]','author_raw':1,'category_raw':1,'category_name':'test'},
        {'item_raw':8,'caption':'[13]','author_raw':1,'category_raw':1,'category_name':'test'},
        {'item_raw':9,'caption':'[77]','author_raw':1,'category_raw':1,'category_name':'test'}]
    for name,rows in [('search',search),('recommendation',rec),('metadata',metadata)]:
        pq.write_table(pa.Table.from_pylist(rows),canonical/f'{name}.parquet')
    write_json(canonical/'users.json',[1])
    write_json(canonical/'manifest.json',{'data_hash':'fixture','cohort_hash':'fixture','cutoffs_utc8_ms':CUTOFFS,
        'files':{name:digest(canonical/name) for name in ('users.json','search.parquet','recommendation.parquet','metadata.parquet')}})
    first=build(canonical,tmp_path/'first');second=build(canonical,tmp_path/'second')
    assert first['data_hash']==second['data_hash']
    ds=PreparedDataset(tmp_path/'first')
    assert 9 not in ds.item_map and 77 not in ds.token_map and 99 not in ds.token_map
    val=ds.requests('validation','search')[0]
    assert val['item_ids'][0]==1 and val['caption_ids'][0]==[1] and val['query_ids']==[1]
    assert [b['timestamp_ms'] for b in val['history']]==[t-2000,t-1000]
    assert val['history'][1]['rec_item_ids']==[] # search result item IDs excluded
    assert all(b['timestamp_ms']<val['timestamp_ms'] for b in val['history'])
