"""Compare frozen PHEE epoch1 weights with upstream and constrained decoding."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_text2event_guarded_pilot import write,digest,counts_score,supervise


def raw_pairs(sequence):
    """Count completed raw leaves for repetition diagnostics, never partial F1."""
    return [tuple(body.strip().split(' ',1)) for body in re.findall(r'<extra_id_0>([^<>]+)<extra_id_1>',sequence)
            if ' ' in body.strip()]


def derive_limit(rows,records_fn):
    """Derive flat multiplicity per source occurrence from TRAIN annotations only."""
    limit=1
    for row in rows:
        pairs=Counter()
        for record in records_fn(row):
            pairs[record['type'],record['trigger']]+=1;pairs.update(record['roles'])
        for (_,surface),number in pairs.items():
            words=surface.split(' ');tokens=row['tokens']
            capacity=sum(tokens[i:i+len(words)]==words for i in range(len(tokens)-len(words)+1))
            if not capacity:raise ValueError('TRAIN mention absent from input')
            limit=max(limit,math.ceil(number/capacity))
    return limit


def worker(a):
    """Verify checkpoint/gold coverage and run both decoders without training."""
    import torch
    source=a.source.resolve();output=a.output
    def event(phase,**fields):
        value={'phase':phase,'time':time.time(),**fields};write(output/'activity.json',value);print(json.dumps(value),flush=True)
    event('PREFLIGHT')
    lock=json.loads(a.lock.read_text());expected_sha='a2236cbef33cc9a32378dac528b474a258bf14f598d8f6405a1b746d0d5463ef'
    if digest(a.checkpoint)!=expected_sha:raise ValueError('Pinned PHEE checkpoint hash mismatch')
    if digest(source/'package-manifest.json')!=lock['source_manifest_sha256']:raise ValueError('Source manifest mismatch')
    for name,h in json.loads((source/'package-manifest.json').read_text()).items():
        if digest(source/name)!=h:raise ValueError('Frozen runtime changed: '+name)
    sys.path.insert(0,str(source))
    from src.baselines.text2event.training import load_components
    from src.baselines.text2event.upstream_decoding import UpstreamConstraint
    from src.baselines.text2event.curriculum import leaf_set
    from src.baselines.text2event.convert import records,convert_substructure
    from src.baselines.text2event.logit_rules import EmptyRootPenalty
    import importlib.util
    spec=importlib.util.spec_from_file_location('phee_diagnostic_constraint',a.decoder)
    diagnostic=importlib.util.module_from_spec(spec);sys.modules[spec.name]=diagnostic;spec.loader.exec_module(diagnostic)
    TreeConstraint=diagnostic.TreeConstraint
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=False,mmap=True)
    run=state['signature']['run'];config=run['config']
    if state['epoch']!=1 or state['signature']['stage']!='substructure' or run['dataset']!='PHEE':raise ValueError('Wrong checkpoint stage/dataset')
    if config!=lock['config'] or state['scheduler']['last_epoch']!=179:raise ValueError('Wrong config/scheduler update')
    for name,h in run['files'].items():
        if digest(source/'bundle'/name)!=h:raise ValueError('Checkpoint bundle mismatch')
    folder=source/'bundle/PHEE'
    rows={s:list(map(json.loads,(folder/(s+'.gold.jsonl')).read_text().splitlines())) for s in ['train','dev']}
    quota=derive_limit(rows['train'],records)
    selected={'train':[rows['train'][i] for i in sorted(random.Random(421).sample(range(len(rows['train'])),8))],
              'dev':rows['dev'][:8]}
    write(output/'request.json',{'trained':False,'test_used':False,'stage':'substructure','checkpoint_epoch':1,
        'checkpoint_sha256':expected_sha,'schema':str(folder/'event.schema'),'flat_repeat_limit':quota,
        'constrained_policy':'source-occurrence','allow_empty_tree':True,'root_penalty':0,
        'selected_ids':{s:[r['id'] for r in rs] for s,rs in selected.items()},
        'selection':'TRAIN random seed421; DEV first8 reproduces observed failures, not representative full DEV',
        'scope':'Combined original-token boundaries, TRAIN quota and completion budget; no isolated attribution'})
    event('MODEL_LOAD');model,t=load_components(config,source/'bundle/tokenizer');model.load_state_dict(state['model'],strict=True);del state
    model.to('cuda').eval();versions=[p._version for p in model.parameters()]
    prefix=config['substructure_source_prefix'];prefix_len=len(t.encode(prefix,add_special_tokens=False));schema=folder/'event.schema'
    def constraint_for(mode,sample,encoded):
        if mode=='upstream':return UpstreamConstraint(t,schema,encoded[prefix_len:],flat=True)
        constraint=TreeConstraint(t,schema,encoded,original_tokens=sample['tokens'],flat=True,
            flat_repeat_limit=quota,max_new_tokens=config['generation_max_new_tokens'],
            repetition_policy='source-occurrence',allow_empty_tree=True)
        if constraint.repetition_policy!='source-occurrence' or constraint.flat_repeat_limit!=quota:raise ValueError('Quota not active')
        return constraint
    coverage={};failures=[]
    for mode in ['upstream','constrained']:
        coverage[mode]={}
        for split,samples in selected.items():
            passed=0
            for sample in samples:
                event('GOLD_COVERAGE',mode=mode,split=split,id=sample['id'])
                encoded=t.encode(prefix+' '.join(sample['tokens']));constraint=constraint_for(mode,sample,encoded);generated=[t.pad_token_id]
                try:
                    for position,token in enumerate(t.encode(convert_substructure(sample)['event'])):
                        if token not in constraint(0,torch.tensor(generated)):raise ValueError('Gold masked at '+str(position))
                        generated.append(token)
                    passed+=1
                except (ValueError,KeyError) as error:failures.append({'mode':mode,'split':split,'id':sample['id'],'reason':str(error)})
            coverage[mode][split]={'checked':len(samples),'passed':passed}
    write(output/'gold-coverage.json',{'coverage':coverage,'failures':failures,'flat_repeat_limit':quota})
    if failures:raise ValueError('Gold coverage failed; no paired generation performed')
    normalize=t.backend_tokenizer.normalizer.normalize_str;results={}
    with torch.no_grad():
        for mode in ['upstream','constrained']:
            results[mode]={}
            for split,samples in selected.items():
                target=output/mode/split;target.mkdir(parents=True);tp=fp=fn=0;issues=[];comparisons=[];started=time.monotonic()
                with (target/'samples.jsonl').open('w') as log:
                    for index,sample in enumerate(samples,1):
                        event('SAMPLE_START',mode=mode,split=split,index=index,total=len(samples),id=sample['id'])
                        encoded=t(prefix+' '.join(sample['tokens']),return_tensors='pt',truncation=False)
                        c=constraint_for(mode,sample,encoded['input_ids'][0].tolist())
                        issue={'id':sample['id'],'status':'ok','decoder':mode,'policy':c.repetition_policy,
                               'flat_repeat_limit':quota if mode=='constrained' else None,'allow_empty_tree':True}
                        predicted=set()
                        try:
                            ids=model.generate(**{k:v.to('cuda') for k,v in encoded.items()},max_new_tokens=config['generation_max_new_tokens'],
                                num_beams=1,do_sample=False,use_cache=True,prefix_allowed_tokens_fn=c,
                                logits_processor=[EmptyRootPenalty(t.convert_tokens_to_ids('<extra_id_0>'),t.convert_tokens_to_ids('<extra_id_1>'),t.pad_token_id,0.)])[0].tolist()
                            text=t.decode([i for i in ids if i not in {t.pad_token_id,t.eos_token_id}],skip_special_tokens=False,clean_up_tokenization_spaces=False)
                            issue.update(generated=text,generated_tokens=len(ids)-1,budget_constrained=c.budget_constrained)
                            if t.eos_token_id not in ids:raise ValueError('generation_limit_without_eos')
                            predicted=leaf_set(text,normalize)
                        except (ValueError,KeyError) as error:issue.update(status='invalid_prediction',reason=str(error))
                        gold=leaf_set(convert_substructure(sample)['event'],normalize)
                        tp+=len(predicted&gold);fp+=len(predicted-gold);fn+=len(gold-predicted)
                        counts=Counter((label,normalize(mention)) for label,mention in raw_pairs(issue.get('generated','')))
                        comparison={'id':sample['id'],'gold_pairs':sorted(gold),'predicted_pairs':sorted(predicted),
                            'matching_pairs':sorted(predicted&gold),'raw_closed_leaves':sum(counts.values()),
                            'repeated_raw_leaves':sum(n-1 for n in counts.values()),'raw_prefix_scored':False}
                        issues.append(issue);comparisons.append(comparison)
                        log.write(json.dumps({'diagnostic':issue,'comparison':comparison},ensure_ascii=False)+'\n');log.flush()
                        event('SAMPLE_DONE',mode=mode,split=split,index=index,status=issue['status'])
                summary={'samples':len(samples),'scores':counts_score(tp,fp,fn),'seconds':time.monotonic()-started,
                    'invalid':sum(r['status']!='ok' for r in issues),
                    'true_empty':sum(r['status']=='ok' and r.get('generated','').replace(' ','')=='<extra_id_0><extra_id_1>' for r in issues),
                    'matching_pairs':tp,'raw_closed_leaves':sum(r['raw_closed_leaves'] for r in comparisons),
                    'repeated_raw_leaves':sum(r['repeated_raw_leaves'] for r in comparisons),
                    'budget_constrained_samples':sum(r.get('budget_constrained',False) for r in issues)}
                write(target/'summary.json',summary);write(target/'diagnostics.json',issues);write(target/'comparisons.json',comparisons)
                results[mode][split]=summary
    if versions!=[p._version for p in model.parameters()]:raise ValueError('Weights changed during replay')
    write(output/'comparison.json',{'results':results,'weights_unchanged':True,'trained':False,'test_used':False})
    write(output/'status.json',{'status':'replay_complete','baseline_complete':False,'trained':False,'test_used':False,
        'weights_unchanged':True,'unique_samples':16,'decoder_sample_evaluations':32})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['source','lock','output','checkpoint','decoder']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--sample-stall-seconds',type=int,default=120);p.add_argument('--progress-stall-seconds',type=int,default=300)
    p.add_argument('--io-stall-seconds',type=int,default=900);p.add_argument('--worker',action='store_true');a=p.parse_args()
    if a.worker:worker(a)
    else:supervise(a,worker_script=__file__)
