"""Evaluate a fixed small TRAIN/DEV subset from a trusted checkpoint, without training."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time
import torch
from src.baselines.text2event.training import load_components
from src.baselines.text2event.curriculum import evaluate_substructure
from src.baselines.text2event.evaluation import evaluate_model
from src.baselines.text2event.evaluation_trace import EvaluationTrace


def select_samples(rows, limit):
    indices=sorted(random.Random(421).sample(range(len(rows)),min(limit,len(rows))))
    return [rows[i] for i in indices]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--expected-checkpoint-epoch',type=int)
    parser.add_argument('--split',choices=['train','dev'],default='dev')
    parser.add_argument('--limit',type=int,choices=[8,16],default=8)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--decoder',choices=['checkpoint','project-constrained','upstream-compatible'],default='checkpoint')
    parser.add_argument('--disallow-empty-tree',action='store_true')
    args=parser.parse_args()
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('GPU unavailable; diagnostic has not started')
    args.output.mkdir(parents=True,exist_ok=False)
    trace=EvaluationTrace(args.output/'trace.jsonl')
    trace.emit('CHECKPOINT_LOAD_START',checkpoint=str(args.checkpoint))
    # best.pt contains weights only; never rewrite or resume the training run.
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    if args.expected_checkpoint_epoch is not None and state['epoch']!=args.expected_checkpoint_epoch:
        raise ValueError('Unexpected checkpoint epoch')
    signature=state['signature'];stage=signature['stage'];run=signature['run']
    if stage not in {'substructure','full'}:raise ValueError('Unsupported checkpoint stage')
    dataset=run['dataset'];config=run['config'];folder=args.bundle/dataset
    checkpoint_config=config
    if args.decoder!='checkpoint':
        import copy
        from scripts.compare_text2event_checkpoints import decoder_config, flat_limit
        config=decoder_config(copy.deepcopy(config),args.decoder)
        if args.decoder=='project-constrained':
            train=list(map(json.loads,(folder/'train.gold.jsonl').read_text().splitlines()))
            config['constraint_repetition_limits'][dataset]['flat_repeat_limit']=flat_limit(train)
    if args.disallow_empty_tree:
        if config.get('substructure_constraint_decoder' if stage=='substructure' else 'constraint_decoder')!='project-constrained':
            raise ValueError('Nonempty-tree diagnostic requires project-constrained decoder')
        import copy
        config=copy.deepcopy(config)
        config['constraint_allow_empty_tree']=False
    for name,digest in run['files'].items():
        with (args.bundle/name).open('rb') as handle:
            if hashlib.file_digest(handle,'sha256').hexdigest()!=digest:
                raise ValueError('Checkpoint bundle differs: '+name)
    rows=list(map(json.loads,(folder/f'{args.split}.gold.jsonl').read_text().splitlines()))
    # Selection uses only a fixed seed and row indices, never model scores/gold difficulty.
    samples=select_samples(rows,args.limit)
    request={'kind':'fixed-checkpoint-'+args.split.upper()+'-runtime-diagnostic','split':args.split,'trained':False,'test_used':False,
             'dataset':dataset,'stage':stage,'checkpoint_epoch':state['epoch'],
             'checkpoint_bytes':args.checkpoint.stat().st_size,'samples':len(samples),
             'ids':[row['id'] for row in samples],'selection_seed':421,
             'decoder_override':args.decoder,'checkpoint_config':checkpoint_config,
             'config':config,'generation_limit_unchanged':config['generation_max_new_tokens']}
    (args.output/'request.json').write_text(json.dumps(request,ensure_ascii=False,indent=2))
    trace.emit('MODEL_LOAD_START',stage=stage,checkpoint_epoch=state['epoch'])
    model,tokenizer=load_components(config,args.bundle/'tokenizer')
    model.load_state_dict(state['model']);del state
    model.to(args.device);model.eval()
    trace.emit('EVALUATION_START',samples=len(samples),stage=stage)
    began=time.perf_counter()
    if stage=='substructure':
        scores,issues=evaluate_substructure(model,tokenizer,samples,folder/'event.schema',config,args.device,observer=trace)
    else:
        scores,predictions,issues=evaluate_model(model,tokenizer,samples,folder/'event.schema',config,args.device,observer=trace)
        (args.output/'predictions.json').write_text(json.dumps(predictions,ensure_ascii=False))
    if stage=='substructure':
        from src.baselines.text2event.convert import convert_substructure, parse
        normalize=lambda s:' '.join(tokenizer.backend_tokenizer.normalizer.normalize_str(s).split())
        comparisons=[]
        for sample,issue in zip(samples,issues):
            gold=parse(convert_substructure(sample)['event'])
            predicted=parse(issue['generated']) if issue['status']=='ok' else []
            pair=lambda rows:{(r['type'],normalize(r['trigger'])) for r in rows}
            comparisons.append({'id':sample['id'],'status':issue['status'],'gold_leaves':gold,
                'predicted_leaves':predicted,'gold_leaf_count':len(gold),'predicted_leaf_count':len(predicted),
                'exact_pair_set_match':pair(gold)==pair(predicted),'matching_pairs':sorted(pair(gold)&pair(predicted))})
        (args.output/'comparisons.json').write_text(json.dumps(comparisons,ensure_ascii=False,indent=2))
    result={'scores':scores,'seconds':time.perf_counter()-began,
            'samples':len(samples),'invalid':sum(x['status']!='ok' for x in issues),
            'true_empty':sum(x['status']=='ok' and x.get('generated','').replace(' ','')=='<extra_id_0><extra_id_1>' for x in issues),
            'timing_totals':{name:sum(x['timing'].get(name,0) for x in issues)
                for name in ['encode','constraint','generate','callback_seconds','sample_seconds']},
            'note':'callback_seconds is included in generate; do not add them together'}
    (args.output/'diagnostics.json').write_text(json.dumps(issues,ensure_ascii=False,indent=2))
    (args.output/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    trace.emit('EVALUATION_COMPLETE',**result)


if __name__=='__main__':main()
