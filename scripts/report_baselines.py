"""Reevaluate saved predictions and verify matched candidates before reporting."""
from __future__ import annotations
from collections import Counter
import json
from pathlib import Path
import pyarrow.parquet as pq
from src.common import ARTIFACTS_ROOT,DATA_ROOT,REPORTS_ROOT,digest,object_hash,read_json,reports_dir,write_json
from src.evaluation.evaluate import evaluate_predictions
from scripts.run_baselines import TABULAR,NEURAL

def alignment(rows):
    result={}
    for row in rows:
        task=row['task'];user=str(row['user_raw'])
        base=':'.join(str(row['request_id']).split(':')[:3])
        if task=='search':
            key=(task,user,base)
            value=(tuple(map(str,row['item_raw'])),tuple(row['labels']),row['cohort'])
        else:
            key=(task,user,base,str(row['item_raw']))
            value=(row['label'],row['cohort'])
        assert key not in result,('duplicate prediction key',key)
        result[key]=value
    return result

def generate(output_dir:Path|None=None):
    output_dir=Path(output_dir) if output_dir is not None else reports_dir()
    output_dir.mkdir(parents=True,exist_ok=True)
    prepared=read_json(DATA_ROOT/'prepared/pilot-v1/manifest.json')
    request_file=DATA_ROOT/'prepared/pilot-v1/requests.parquet'
    assert digest(request_file)==prepared['files']['requests.parquet']
    requests=pq.read_table(request_file,columns=['user_raw','request_id','task','item_raw','labels','cohort'],
        filters=[('split','=','validation')]).to_pylist()
    authoritative=[]
    for row in requests:
        if row['task']=='search':authoritative.append(row)
        else:
            authoritative.extend({**row,'item_raw':item,'label':label} for item,label in zip(row['item_raw'],row['labels'],strict=True))
    reference=alignment(authoritative);results=[]
    ledger=[json.loads(line) for path in sorted(REPORTS_ROOT.glob('*/experiment-ledger.jsonl'))
        for line in path.read_text().splitlines() if line]
    for run_id,_,_ in (*TABULAR,*NEURAL):
        directory=ARTIFACTS_ROOT/'runs'/run_id
        neural=run_id in {r[0] for r in NEURAL}
        config=read_json(directory/('config.json' if neural else 'manifest.json'))
        assert config['data_hash']==prepared['data_hash'],f'{run_id}: wrong data snapshot'
        report=read_json(directory/'metrics.json')
        if neural:
            assert report['status']=='complete' and report['completed_steps']==1000
        filename='validation-best.jsonl' if neural else 'validation_predictions.jsonl'
        rows=[json.loads(line) for line in (directory/filename).read_text().splitlines()]
        assert all(row.get('fallback') is False for row in rows)
        current=alignment(rows)
        assert current==reference,f'{run_id}: prepared validation candidate/label/cohort mismatch'
        metrics=evaluate_predictions(rows)
        selected=read_json(directory/'validation-best.json') if neural else report
        assert metrics['search']==selected['search'] and metrics['recommendation']==selected['recommendation']
        attempts=[e for e in ledger if e.get('run_id')==run_id and e.get('status') in ('complete','time_budget_reached','profile_complete')]
        wall=sum(e.get('call_wall_seconds',e.get('wall_time_seconds',0)) for e in attempts) if neural else config['wall_time_seconds']
        result={'run_id':run_id,'data_hash':prepared['data_hash'],
            'config_hash':config.get('config_hash') if neural else object_hash({k:config.get(k) for k in ('model','seed','model_config','feature_policy','row_cap','candidate_cap','candidate_policy')}),
            'config_artifact_sha256':digest(directory/('config.json' if neural else 'manifest.json')),
            'prediction_sha256':digest(directory/filename),'quality':metrics,'wall_seconds':wall,
            'optimizer_updates':report.get('completed_steps') if neural else None,
            'selected_step':selected['step'] if neural else None,'peak_rss_bytes':report.get('peak_rss_bytes_observed') if neural else config.get('peak_memory_bytes'),
            'parameters':report.get('parameter_count') if neural else None,
            'parameter_bytes_fp32':report.get('parameter_bytes_fp32') if neural else None,
            'selected_joint_log_loss':selected.get('joint_log_loss') if neural else None,
            'provenance':report.get('provenance') if neural else {k:config.get(k) for k in ('code_commit','code_content_hash','environment_lock')},
            'candidate_alignment_verified':True}
        results.append(result)
    write_json(output_dir/'baseline-results.json',{'status':'verified','scope':'six single-seed validation screens; all candidates and labels aligned; metrics recomputed from saved predictions',
        'data_hash':prepared['data_hash'],'authoritative_request_sha256':digest(request_file),
        'validation_request_counts':dict(Counter(row['task'] for row in requests)),
        'validation_prediction_counts':dict(Counter(row['task'] for row in authoritative)),
        'results':results})
    lines=['# TRACE development baselines','',
        'Six predeclared screens on the same 2,000-user training cohort and validation candidates. These are development references, not a convergence, SOTA, or final novelty claim. Test predictions have not been generated.','',
        '| Run | Search user-macro NDCG@10 | Transition NDCG@10 | Rec AP | Rec log loss | Selected step | Wall minutes |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in results:
        q=r['quality'];s=q['search'];rec=q['recommendation']
        lines.append(f"| {r['run_id']} | {s['overall']['user_macro']['ndcg@10']:.4f} | {s['transition']['user_macro']['ndcg@10']:.4f} | {rec['average_precision']:.4f} | {rec['log_loss']:.4f} | {r['selected_step'] if r['selected_step'] is not None else 'n/a'} | {r['wall_seconds']/60:.2f} |")
    lines.extend(['','## Measured training resources','',
        '| Run | Recorded process memory (GiB) | Learned parameters | FP32 parameter bytes (MiB) |',
        '|---|---:|---:|---:|'])
    for r in results:
        rss=f"{r['peak_rss_bytes']/2**30:.2f}" if r['peak_rss_bytes'] is not None else 'not recorded'
        params=f"{r['parameters']:,}" if r['parameters'] is not None else 'n/a'
        size=f"{r['parameter_bytes_fp32']/2**20:.2f}" if r['parameter_bytes_fp32'] is not None else 'n/a'
        lines.append(f"| {r['run_id']} | {rss} | {params} | {size} |")
    lines.extend(['',
        'Hardware, thread count and software provenance are stored with each run. Memory is the recorded training-process working set, not a serving measurement; neural RSS samples can miss instantaneous peaks. Tabular wall time includes prepared-dataset construction, feature extraction, fitting and validation. Neural wall time covers the recorded trainer call, including validation/checkpointing, and excludes initial PreparedDataset construction. FP32 parameter bytes exclude optimizer states and serialization overhead; resumable best.pt files include optimizer state and are not deployment artifacts. These are preliminary resource measurements, with serving latency deferred.'])
    strongest_separate=min((r for r in results if r['run_id'].startswith('v1-separate')),key=lambda r:read_json(ARTIFACTS_ROOT/'runs'/r['run_id']/'validation-best.json')['joint_log_loss'])
    best_search=max(results,key=lambda r:r['quality']['search']['overall']['user_macro']['ndcg@10'])
    shared=next(r for r in results if r['run_id']=='v1-shared-lr001')
    lines.extend(['','## Contracts and selection','',
        'Prediction keys, candidate IDs/order, labels, and cohort flags for all six models match the hashed prepared validation requests. Metrics were recomputed from saved prediction files and match the stored evaluator results. Config/data/prediction hashes and full metric denominators are in baseline-results.json. Missing/OOV candidates are retained; all saved fallback flags are false because these models score those inputs directly.','',
        'The two separate-encoder learning rates were tuned under the same 1,000-update budget and validation frequency. Selected separate configuration: **'+strongest_separate['run_id']+'**, using minimum equal-task validation BCE. Shared checkpoint selection uses the same rule. Baseline 001 fixes the allocation and rule.','',
        '## Preliminary observations','',
        f"The strongest overall search score in this screen belongs to {best_search['run_id']} ({best_search['quality']['search']['overall']['user_macro']['ndcg@10']:.4f}). Shared neural search ({shared['quality']['search']['overall']['user_macro']['ndcg@10']:.4f}) has not surpassed that control. Both neural approaches improve recommendation AP over the cheap/tabular controls in this single validation screen.", '',
        f"Shared recommendation AP is {shared['quality']['recommendation']['average_precision']:.4f}, compared with {strongest_separate['quality']['recommendation']['average_precision']:.4f} for the validation-selected separate model. Shared uses half as many parameters and its recorded trainer call is shorter. These observations require longer matched training and repeated seeds before making a quality/efficiency claim.", '',
        'Use feature-diagnostics.md to inspect frozen-model query/history/caption sensitivities. Investigate optimization, query support and OOV coverage before claiming personalized search improvement. Compare metadata-enabled logistic and ID-only controls to assess sensitivity.','',
        '## What these results can establish','',
        'The pipeline supports end-to-end exposure learning and candidate ranking; cheap/tabular baselines are meaningful controls. More complex models must earn their additional cost on these comparisons. Differences are single-seed validation observations and should not be interpreted as confirmed gains.','',
        'Neural training presents 32,000 search groups and 32,000 recommendation exposures, with replacement. The recommendation sample budget covers substantially less than one full exposure pass, so these results do not establish convergence. The shared model has joint identities; separate models filter the same last-50-block observable window to their own service and recompute gaps. This controls the available window, rather than giving each separate model 50 service-specific blocks.','',
        f"Transition validation ranking uses {shared['quality']['search']['transition']['eligible_requests']} requests from {shared['quality']['search']['transition']['eligible_users']} users. Overall search ranking uses {shared['quality']['search']['overall']['eligible_requests']} requests from {shared['quality']['search']['overall']['eligible_users']} users. Exclusions remain in pointwise/coverage reporting. Recommendation includes {shared['quality']['recommendation']['count']:,} exposures with prevalence {shared['quality']['recommendation']['prevalence']:.4f}. Item OOV is {prepared['counts']['validation']['oov_items']/prepared['counts']['validation']['exposures']:.1%} across validation candidates/exposures. The static-caption assumption is tested with logistic ID-only sensitivity.",'',
        'Cheap scores use exposure frequency and token overlap with a training-only global task intercept; they are not calibrated click probabilities. Logistic models use train-fitted scaling and equal-task/group/exposure weights. There are no unobserved synthetic negatives, label-derived CTR features, current labels in inputs, or historical outcomes.','',
        '## Next checkpoint','',
        'Build the attributed clean-protocol joint reference (checkpoint 4), then measure whether it is a useful teacher (checkpoint 5). Revisit training budget and transition support before any distillation claim. Use measured throughput to decide cloud runs; retain these baselines and their candidate protocol.'])
    (output_dir/'baselines.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Six baseline comparisons verified and reported',flush=True)

if __name__=='__main__':generate()
