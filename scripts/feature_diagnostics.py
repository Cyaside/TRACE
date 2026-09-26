"""Fixed-model validation masking diagnostics; these are not retraining controls."""
from collections.abc import Sequence
from dataclasses import fields
import torch
from src.common import ARTIFACTS_ROOT,DATA_ROOT,digest,read_json,reports_dir,write_json
from src.features.dataset import PreparedDataset
from src.training.neural import TrainConfig,build_models,validation_metrics

class MaskedRequests(Sequence):
    def __init__(self,source,mode):self.source,self.mode=source,mode
    def __len__(self):return len(self.source)
    def __getitem__(self,index):
        if isinstance(index,slice):return [self[i] for i in range(*index.indices(len(self)))]
        row=dict(self.source[index])
        if self.mode=='no_current_query':row['query_ids']=[];row['query_tokens']=[]
        if self.mode=='no_history':row['history']=[];row['history_ids']=[];row['request_gap']=0
        if self.mode=='no_caption':row['caption_ids']=[[] for _ in row['item_ids']]
        return row

if __name__=='__main__':
    torch.set_num_threads(4)
    ds=PreparedDataset(DATA_ROOT/'prepared/pilot-v1')
    directory=ARTIFACTS_ROOT/'runs/v1-shared-lr001'
    output=reports_dir();output.mkdir(parents=True,exist_ok=True)
    cfg=read_json(directory/'config.json')
    assert cfg['data_hash']==ds.data_hash
    config=TrainConfig(**{k:cfg[k] for k in {f.name for f in fields(TrainConfig)} if k in cfg})
    model=build_models(ds.item_count,ds.token_count,config)
    snapshot=torch.load(directory/'best.pt',map_location='cpu',weights_only=False)
    assert snapshot['data_hash']==ds.data_hash and snapshot['config_hash']==cfg['config_hash']
    model.load_state_dict(snapshot['models']);del snapshot
    baseline=read_json(directory/'validation-best.json')
    source=ds.requests('validation');results={}
    for mode in ('no_current_query','no_history','no_caption'):
        metrics,_=validation_metrics(model,MaskedRequests(source,mode),'cpu',32)
        results[mode]=metrics
        print(mode,metrics['joint_log_loss'],flush=True)
    write_json(output/'feature-diagnostics.json',{'data_hash':ds.data_hash,'run_id':'v1-shared-lr001',
        'checkpoint_sha256':digest(directory/'best.pt'),'baseline':baseline,'masked':results,
        'interpretation':'Frozen-model masking sensitivity on validation; distribution-shifting diagnostic, not retrained ablation or causal feature effect'})
    lines=['# Feature contribution diagnostics','',
        'The selected shared checkpoint is frozen. Each diagnostic changes one permitted input channel and preserves all validation candidates/labels. No additional model is trained and no test rows are scored.','',
        '| Input policy | Search user-macro NDCG@10 | Rec AP | Rec log loss |',
        '|---|---:|---:|---:|']
    for name,metrics in [('original',baseline),*results.items()]:
        lines.append(f"| {name} | {metrics['search']['overall']['user_macro']['ndcg@10']:.4f} | {metrics['recommendation']['average_precision']:.4f} | {metrics['recommendation']['log_loss']:.4f} |")
    lines.extend(['','These diagnose the trained model\'s dependence on current query, prior identity history, and candidate caption channels. Masking changes the input distribution; it does not establish the quality of a model retrained without that channel. The separately trained logistic ID-only model is the metadata-snapshot sensitivity control.','',
        'The model has no user-ID embedding, historical outcome inputs, or historical search-result item IDs. Inspect the prepared validation OOV coverage and evaluator cohort denominators when interpreting these diagnostics. If a removed channel has little effect or helps a metric, investigate data/optimization and preserve that finding before expanding architecture.'])
    (output/'feature-diagnostics.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
