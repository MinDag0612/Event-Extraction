"""Bounded two-stage TRAIN pilot with root-choice probes; not baseline results."""
import argparse
import json
import random
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from src.baselines.text2event.training import load_components, BatchCollator, train_epoch
from src.baselines.text2event.convert import convert, convert_substructure
from src.baselines.text2event.evaluation import evaluate_model
from src.baselines.text2event.curriculum import evaluate_substructure
from src.baselines.text2event.diagnostics import generation_summary


@torch.no_grad()
def root_choices(model, tokenizer, samples, prefix, device):
    model.eval();output=[]
    op=tokenizer.convert_tokens_to_ids('<extra_id_0>');cl=tokenizer.convert_tokens_to_ids('<extra_id_1>')
    for row in samples:
        source=tokenizer(prefix+' '.join(row['tokens']),return_tensors='pt',truncation=False)
        logits=model(**{k:v.to(device) for k,v in source.items()},
            decoder_input_ids=torch.tensor([[tokenizer.pad_token_id,op]],device=device)).logits[0,-1]
        probs=logits[[op,cl]].float().softmax(0).tolist()
        output.append({'id':row['id'],'has_gold_event':bool(row['events']),
                       'p_open_given_open_or_close':probs[0],'p_close_given_open_or_close':probs[1],
                       'close_minus_open_logit':float(logits[cl]-logits[op])})
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--dataset',choices=['BKEE','GENEVA','PHEE','VHE'],required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--examples',type=int,default=16)
    p.add_argument('--epochs-per-stage',type=int,default=3)
    p.add_argument('--device',default='cuda')
    p.add_argument('--compare-penalties',action='store_true',help='Train/select with zero penalty, compare fixed final weights')
    a=p.parse_args()
    if a.examples<2 or a.epochs_per_stage<1:p.error('Invalid pilot size')
    a.output.mkdir(parents=True,exist_ok=False)
    config=json.loads(Path('configs/text2event/preparation.json').read_text())
    if a.compare_penalties:
        config['empty_root_penalty']=0.  # Keep stage selection independent of penalty tuning.
    random.seed(421);torch.manual_seed(421)
    rows=list(map(json.loads,(a.bundle/a.dataset/'train.gold.jsonl').read_text().splitlines()))
    positives=sorted([r for r in rows if r['events']],key=lambda r:len(r['tokens']))
    negatives=sorted([r for r in rows if not r['events']],key=lambda r:len(r['tokens']))
    nneg=min(a.examples//4,len(negatives));selected=positives[:a.examples-nneg]+negatives[:nneg]
    if len(selected)!=a.examples:raise ValueError('Not enough samples')
    # Evaluate positive examples plus a negative when available; all are TRAIN.
    probes=selected[:3]+(selected[-1:] if nneg else selected[3:4])
    (a.output/'selected-train.json').write_text(json.dumps(selected,ensure_ascii=False)+'\n')
    model,t=load_components(config,a.bundle/'tokenizer');model.to(a.device)
    model.gradient_checkpointing_enable();model.config.use_cache=False
    (a.output/'model-loading.json').write_text(json.dumps(model.text2event_load_report)+'\n')
    for stage,converter,prefix in [('substructure',convert_substructure,config['substructure_source_prefix']),
                                   ('full',convert,config['full_source_prefix'])]:
        examples=[]
        for row in selected:
            record=converter(row);source=t(prefix+record['text'],truncation=False)
            target=t(record['event'],truncation=False)['input_ids']
            if len(source['input_ids'])>config['max_source_length'] or len(target)>config['max_target_length']:
                raise ValueError('Pilot length overflow')
            examples.append(dict(id=row['id'],input_ids=source['input_ids'],attention_mask=source['attention_mask'],labels=target))
        loader=DataLoader(examples,batch_size=1,shuffle=True,collate_fn=BatchCollator(t))
        opt=torch.optim.AdamW(model.parameters(),lr=5e-5,weight_decay=0.)
        sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.)
        best=-1.
        for epoch in range(a.epochs_per_stage):
            began=time.monotonic();print(f'{stage} epoch {epoch+1} TRAIN start',flush=True)
            metrics=train_epoch(model,loader,opt,sched,a.device,progress_every=4)
            metrics['train_seconds']=time.monotonic()-began
            # Weights only: recover learned weights if evaluation times out.
            # No claim of exact optimizer/sampler resume from this diagnostic.
            temporary=a.output/'latest-weights.tmp'
            torch.save({'model':model.state_dict(),'stage':stage,'epoch':epoch+1,'kind':'weights-only-pilot'},temporary)
            temporary.replace(a.output/'latest-weights.pt')
            print(f'{stage} epoch {epoch+1} WEIGHTS saved; EVAL start',flush=True)
            metrics['root_choices']=root_choices(model,t,probes,prefix,a.device)
            (a.output/f'{stage}-{epoch+1}-root.json').write_text(json.dumps(metrics['root_choices'],indent=2)+'\n')
            print(json.dumps({'stage':stage,'epoch':epoch+1,'root_choices':metrics['root_choices']}),flush=True)
            evaluated=time.monotonic()
            if stage=='substructure':
                score,issues=evaluate_substructure(model,t,probes,a.bundle/a.dataset/'event.schema',config,a.device)
                value=score['f1']
            else:
                score,pred,issues=evaluate_model(model,t,probes,a.bundle/a.dataset/'event.schema',config,a.device)
                metrics['generation']=generation_summary(pred,issues);value=score['AC']['f1']
            metrics.update(stage=stage,epoch=epoch+1,scores=score,diagnostics=issues,eval_seconds=time.monotonic()-evaluated)
            (a.output/f'{stage}-{epoch+1}.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2)+'\n')
            if stage=='substructure' and value>best:
                best=value;torch.save(model.state_dict(),a.output/'best-substructure.pt')
            print(json.dumps({k:v for k,v in metrics.items() if k!='diagnostics'}),flush=True)
        if stage=='substructure':
            model.load_state_dict(torch.load(a.output/'best-substructure.pt',map_location='cpu',weights_only=True))
            (a.output/'best-substructure.pt').unlink()
    if a.compare_penalties:
        comparison={}
        variants=[('upstream-0','upstream-compatible',0.),
                  ('custom-0','project-constrained',0.),
                  ('custom-1','project-constrained',1.)]
        for name,decoder,penalty in variants:
            trial=dict(config,constraint_decoder=decoder,empty_root_penalty=penalty)
            score,pred,issues=evaluate_model(model,t,probes,a.bundle/a.dataset/'event.schema',trial,a.device)
            positive=[(g,p) for g,p in zip(probes,pred) if g['events']]
            negative=[(g,p) for g,p in zip(probes,pred) if not g['events']]
            comparison[name]={'scores':score,'generation':generation_summary(pred,issues),
                'positive_samples':len(positive),'positive_predicted_empty':sum(not p['events'] for _,p in positive),
                'negative_samples':len(negative),'negative_predicted_event':sum(bool(p['events']) for _,p in negative),
                'predictions':pred,'diagnostics':issues,'empty_root_penalty':penalty,'decoder':decoder,
                'probe_ids':[g['id'] for g in probes]}
            (a.output/'comparison.json').write_text(json.dumps(comparison,ensure_ascii=False,indent=2)+'\n')
            print('COMPARISON',name,json.dumps(comparison[name]['generation']),flush=True)
    (a.output/'completion.json').write_text(json.dumps({'kind':'TRAIN-only-short-pilot-not-baseline','dataset':a.dataset,'examples':a.examples,'epochs_per_stage':a.epochs_per_stage})+'\n')


if __name__=='__main__':main()
