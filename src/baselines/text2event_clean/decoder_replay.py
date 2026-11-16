"""Compare two decoders on one frozen epoch2 checkpoint and identical samples."""
from collections import Counter
import copy
import json
import random
import re
import time
import torch
from src.baselines.text2event.training import load_components
from src.baselines.text2event.curriculum import evaluate_substructure
from src.baselines.text2event.convert import convert_substructure, parse, OPEN, CLOSE
from src.baselines.text2event.decoding import TreeConstraint
from src.baselines.text2event.upstream_decoding import UpstreamConstraint
from src.baselines.text2event.evaluation_trace import EvaluationTrace
from src.baselines.text2event_clean.checkpoint_audit import validate_checkpoint
from scripts.compare_text2event_checkpoints import decoder_config, flat_limit


def make_decoder_configs(base, train):
    """Derive repetition limits from TRAIN while preserving model and generation settings."""
    upstream=decoder_config(copy.deepcopy(base),'upstream-compatible')
    constrained=decoder_config(copy.deepcopy(base),'project-constrained')
    constrained['constraint_allow_empty_tree']=True
    constrained['constraint_repetition_limits']['GENEVA']={
        'flat_repeat_limit':flat_limit(train),'event_repeat_limit':1,'argument_repeat_limit':1}
    return {'upstream':upstream,'constrained':constrained}


def complete_flat_pairs(sequence):
    """Extract completed flat leaves from raw text for repetition diagnostics only."""
    parts=re.split('('+re.escape(OPEN)+'|'+re.escape(CLOSE)+')',sequence)
    pairs=[]
    for index,part in enumerate(parts):
        if part==OPEN and index+2<len(parts) and parts[index+2]==CLOSE:
            body=parts[index+1].strip()
            if ' ' in body:
                label,mention=body.split(' ',1)
                if mention.strip():pairs.append((label,mention.strip()))
    return pairs


def gold_coverage(tokenizer,samples,schema,config):
    """Replay gold prefixes against the same constraint construction as evaluation."""
    failures=[]
    for sample in samples:
        prefix=config['substructure_source_prefix']
        encoded=tokenizer(prefix+' '.join(sample['tokens']))['input_ids']
        if config['substructure_constraint_decoder']=='project-constrained':
            constraint=TreeConstraint(tokenizer,schema,encoded,original_tokens=sample['tokens'],
                flat=True,flat_repeat_limit=config['constraint_repetition_limits']['GENEVA']['flat_repeat_limit'],
                max_new_tokens=config['generation_max_new_tokens'],allow_empty_tree=True)
        else:
            length=len(tokenizer.encode(prefix,add_special_tokens=False))
            constraint=UpstreamConstraint(tokenizer,schema,encoded[length:],flat=True)
        ids=[tokenizer.pad_token_id]
        try:
            for pos,token in enumerate(tokenizer.encode(convert_substructure(sample)['event'])):
                if token not in constraint(0,torch.tensor(ids)):
                    raise ValueError(f'Gold token masked at position{pos}')
                ids.append(token)
        except (ValueError,KeyError) as error:failures.append({'id':sample['id'],'reason':str(error)})
    return {'checked':len(samples),'passed':len(samples)-len(failures),'failures':failures}


@torch.no_grad()
def run(bundle,base,checkpoint,output,expected_epoch):
    """Load model once, verify gold coverage, then evaluate paired TRAIN/DEV decoders."""
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    output.mkdir(parents=True,exist_ok=False)
    def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2))
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    validate_checkpoint(state,base,bundle,expected_epoch)
    if state['signature']['run'].get('ablation_branch')!='constant':raise ValueError('Expected the pinned constant-LR branch')
    signature=state['signature']
    model,t=load_components(base,bundle/'tokenizer');model.load_state_dict(state['model']);del state
    model.to('cuda').eval()
    versions=[p._version for p in model.parameters()]
    folder=bundle/'GENEVA'
    rows={split:list(map(json.loads,(folder/f'{split}.gold.jsonl').read_text().splitlines())) for split in ['train','dev']}
    configs=make_decoder_configs(base,rows['train'])
    selected={split:[items[i] for i in sorted(random.Random(421).sample(range(len(items)),8))] for split,items in rows.items()}
    save(output/'request.json',{'kind':'same-weights-paired-decoder-replay','source_signature':signature,
        'checkpoint_epoch':expected_epoch,'trained':False,'test_used':False,'configs':configs,
        'ids':{split:[x['id'] for x in samples] for split,samples in selected.items()},
        'scope':'Combined decoder comparison includes original-token span boundaries, quotas and completion budget.'})
    coverage={name:{split:gold_coverage(t,samples,folder/'event.schema',config)
              for split,samples in selected.items()} for name,config in configs.items()}
    save(output/'gold-coverage.json',coverage)
    if any(r['failures'] for mode in coverage.values() for r in mode.values()):
        raise ValueError('Gold prefix coverage failed; paired generation not started')
    normalize=lambda text:' '.join(t.backend_tokenizer.normalizer.normalize_str(text).split())
    results={}
    for name,config in configs.items():
        results[name]={}
        for split,samples in selected.items():
            target=output/name/split;target.mkdir(parents=True)
            trace=EvaluationTrace(target/'trace.jsonl')
            print(json.dumps({'phase':'DECODER_REPLAY','decoder':name,'split':split,'samples':8}),flush=True)
            began=time.perf_counter()
            with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
                scores,issues=evaluate_substructure(model,t,samples,folder/'event.schema',config,'cuda',observer=trace)
            comparisons=[];repeats=0;total_leaves=0;correct_pairs=0
            for sample,issue in zip(samples,issues):
                raw=complete_flat_pairs(issue.get('generated',''))
                counts=Counter((label,normalize(mention)) for label,mention in raw)
                repeated=sum(n-1 for n in counts.values());repeats+=repeated;total_leaves+=len(raw)
                valid=parse(issue['generated']) if issue['status']=='ok' else []
                gold=parse(convert_substructure(sample)['event'])
                pair=lambda items:{(x['type'],normalize(x['trigger'])) for x in items}
                matching=sorted(pair(valid)&pair(gold));correct_pairs+=len(matching)
                comparisons.append({'id':sample['id'],'status':issue['status'],'gold':gold,'predicted':valid,
                    'matching_pairs':matching,'complete_raw_leaves':len(raw),'repeated_raw_leaves':repeated,
                    'raw_leaves_used_for_scoring':False})
            summary={'scores':scores,'seconds':time.perf_counter()-began,'samples':8,
                'invalid':sum(x['status']!='ok' for x in issues),
                'true_empty':sum(x['status']=='ok' and x['generated'].replace(' ','')==OPEN+CLOSE for x in issues),
                'repeated_raw_leaves':repeats,'complete_raw_leaves':total_leaves,'matching_pairs_in_valid_predictions':correct_pairs,
                'budget_constrained_samples':sum(x.get('budget_constrained',False) for x in issues)}
            save(target/'summary.json',summary);save(target/'diagnostics.json',issues);save(target/'comparisons.json',comparisons)
            results[name][split]=summary
            print(json.dumps({'decoder':name,'split':split,**summary}),flush=True)
    unchanged=versions==[p._version for p in model.parameters()]
    if not unchanged:raise ValueError('Model parameter versions changed during replay')
    save(output/'comparison.json',{'results':results,'weights_unchanged':unchanged,'trained':False,'test_used':False})
    save(output/'completion.json',{'status':'complete','weights_unchanged':True,'decoder_sample_evaluations':32,'test_used':False})
