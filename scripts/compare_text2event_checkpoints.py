"""Paired decoder replay. No training; checkpoint and selected rows stay fixed."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time
import torch
from src.baselines.text2event.convert import records,convert,convert_substructure,parse
from src.baselines.text2event.training import load_components
from src.baselines.text2event.decoding import TreeConstraint
from src.baselines.text2event.curriculum import evaluate_substructure
from src.baselines.text2event.evaluation import evaluate_model
from src.baselines.text2event.diagnostics import generation_summary


def write(path,data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')


def decoder_config(config, mode):
    # The upstream config carries repetition_policy='none'. Changing only the
    # decoder name leaves full-stage quotas disabled: set the policy explicitly.
    return dict(config, constraint_decoder=mode, substructure_constraint_decoder=mode,
                constraint_repetition_policy='source-occurrence' if mode=='project-constrained' else 'none')


def flat_limit(rows):
    """Counts shared flat leaves across events, unlike per-event full quotas."""
    limit=1
    for row in rows:
        pairs=Counter()
        for event in records(row):
            pairs[event['type'],event['trigger']]+=1
            pairs.update(event['roles'])
        surfaces={surface for _,surface in pairs}
        capacity=Counter();max_chars=max(map(len,surfaces),default=0)
        for start in range(len(row['tokens'])):
            text=''
            for token in row['tokens'][start:]:
                text=(text+' '+token) if text else token
                if len(text)>max_chars:break
                if text in surfaces:capacity[text]+=1
        for (_,surface),count in pairs.items():
            if not capacity[surface]:raise ValueError('Gold surface absent from input')
            limit=max(limit,math.ceil(count/capacity[surface]))
    return limit


def coverage(tokenizer,rows,schema,config,flat):
    limits=config['constraint_repetition_limits'][schema.parent.name]
    failures=[]
    for row in rows:
        c=TreeConstraint(tokenizer,schema,[],original_tokens=row['tokens'],flat=flat,
            max_new_tokens=config['generation_max_new_tokens'],
            event_repeat_limit=limits['event_repeat_limit'],argument_repeat_limit=limits['argument_repeat_limit'],
            flat_repeat_limit=limits['flat_repeat_limit'])
        prefix=[tokenizer.pad_token_id]
        try:
            for pos,token in enumerate(tokenizer.encode((convert_substructure if flat else convert)(row)['event'])):
                if token not in c(0,torch.tensor(prefix)):
                    raise ValueError(f'Gold token rejected at {pos}: {token}')
                prefix.append(token)
        except (ValueError,KeyError) as exc:failures.append({'id':row['id'],'reason':str(exc)})
    return {'checked':len(rows),'passed':len(rows)-len(failures),'failures':failures}


def diagnostics(rows,predictions,issues):
    positive={r['id']:bool(r['events']) for r in rows}
    valid_empty=invalid=positive_empty=negative_nonempty=repeat_leaves=0
    for pred,issue in zip(predictions,issues):
        invalid+=issue['status']!='ok'
        valid_empty+=issue['status']=='ok' and not pred['events']
        positive_empty+=positive[pred['id']] and not pred['events']
        negative_nonempty+=not positive[pred['id']] and bool(pred['events'])
        if 'generated' in issue:
            try:
                leaves=Counter((e['type'],e['trigger'],tuple(e['roles'])) for e in parse(issue['generated']))
                repeat_leaves+=sum(n-1 for n in leaves.values())
            except ValueError:pass
    return {'valid_empty_samples':valid_empty,'invalid_samples':invalid,
        'positive_samples':sum(positive.values()),'positive_samples_with_empty_scored_prediction':positive_empty,
        'negative_samples_with_nonempty_prediction':negative_nonempty,'repeated_surface_records':repeat_leaves}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True);p.add_argument('--input',type=Path,required=True)
    p.add_argument('--dataset',choices=['BKEE','GENEVA','PHEE','VHE'],required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    flat=a.dataset=='VHE';folder=a.bundle/a.dataset;source=a.input
    rows=json.loads((source/('selected-train.json' if flat else 'selected-dev.json')).read_text())
    checkpoint=source/('latest-weights.pt' if flat else 'best-full.pt')
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if state.get('stage')!=('substructure' if flat else 'full'):raise ValueError('Unexpected checkpoint stage')
    config=json.loads(Path('configs/text2event/preparation-upstream.json').read_text())
    config['empty_root_penalty']=0.
    train_file=folder/'train.gold.jsonl'
    limit=flat_limit(list(map(json.loads,train_file.read_text().splitlines()))) if flat else 1
    config['constraint_repetition_limits'][a.dataset]['flat_repeat_limit']=limit
    model,t=load_components(config,a.bundle/'tokenizer');model.load_state_dict(state['model']);del state
    model.to(a.device);model.eval()
    # Auditing held-out gold is separate from generation. A failure stops the
    # comparison; it never changes rules to fit individual held-out answers.
    cover=coverage(t,rows,folder/'event.schema',config,flat)
    write(a.output/'gold-prefix-coverage.json',cover)
    if cover['failures']:raise ValueError('Constraint rejects selected gold: review before inference')
    write(a.output/'run.json',{'kind':'paired-checkpoint-decoder-replay','dataset':a.dataset,
        'stage':'substructure' if flat else 'full','split':'TRAIN' if flat else 'DEV','samples':len(rows),
        'checkpoint':str(checkpoint),'checkpoint_bytes':checkpoint.stat().st_size,
        'flat_repeat_limit':limit,'flat_quota_derived_from':'all TRAIN only',
        'train_sha256':hashlib.sha256(train_file.read_bytes()).hexdigest(),
        'penalty':0,'use_cache_explicit':True,'config':config})
    summary={}
    for mode in ['upstream-compatible','project-constrained']:
        current=decoder_config(config,mode)
        started=time.monotonic()
        if flat:
            scores,issues=evaluate_substructure(model,t,rows,folder/'event.schema',current,a.device)
            extra={'invalid_samples':sum(i['status']!='ok' for i in issues)}
        else:
            scores,pred,issues=evaluate_model(model,t,rows,folder/'event.schema',current,a.device)
            write(a.output/f'{mode}-predictions.json',pred)
            extra={**generation_summary(pred,issues),**diagnostics(rows,pred,issues)}
        summary[mode]={'scores':scores,'seconds':time.monotonic()-started,'diagnostics':extra,
                       'repetition_policy':current['constraint_repetition_policy']}
        write(a.output/f'{mode}-diagnostics.json',issues);write(a.output/'comparison.json',summary)
        print(json.dumps({mode:summary[mode]}),flush=True)
    write(a.output/'completion.json',{'trained':False,'compared':list(summary),'test_used':False})


if __name__=='__main__':main()
