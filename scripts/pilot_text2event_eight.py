"""Eight-TRAIN upstream-style diagnostic. No custom rules, no gate, no TEST."""
import argparse
import json
import random
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from scripts.audit_text2event_eight import audit_selected
from src.baselines.text2event.training import load_components,BatchCollator,train_epoch
from src.baselines.text2event.convert import convert,convert_substructure,parse,to_unified
from src.baselines.text2event.curriculum import evaluate_substructure
from src.baselines.text2event.evaluation import evaluate_model,score_by_id
from src.baselines.text2event.diagnostics import generation_summary
from src.baselines.text2event.learning_probe import teacher_probe
from src.baselines.text2event.pilot_utils import fixed_subset,save_weights_atomic
from src.baselines.text2event.early_stopping import EarlyStopping


def write(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')


def encode(rows,converter,prefix,tokenizer,config):
    output=[]
    for row in rows:
        converted=converter(row);source=tokenizer(prefix+converted['text'],truncation=False)
        target=tokenizer(converted['event'],truncation=False)['input_ids']
        if len(source['input_ids'])>config['max_source_length'] or len(target)>config['max_target_length']:
            raise ValueError(f'Length overflow: {row["id"]}')
        output.append(dict(id=row['id'],input_ids=source['input_ids'],attention_mask=source['attention_mask'],labels=target))
    return output


def surface_diagnostics(rows,issues,tokenizer):
    normalize=tokenizer.backend_tokenizer.normalizer.normalize_str
    def canonical(text):
        return sorted((r['type'],normalize(r['trigger']),tuple(sorted((role,normalize(txt)) for role,txt in r['roles']))) for r in parse(text))
    exact=0;roundtrip=[]
    for row,issue in zip(rows,issues):
        target=convert(row)['event']
        if issue['status']=='ok':
            exact+=canonical(target)==canonical(issue['generated'])
        roundtrip.append(to_unified(target,row['id'],row['text'],row['tokens'],
            ambiguity_policy='nearest-trigger',surface_normalizer=normalize))
    return {'samples':len(rows),'surface_tree_exact_samples':exact,
            'gold_target_roundtrip_reference':score_by_id(roundtrip,rows)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--dataset',choices=['BKEE','GENEVA','PHEE','VHE'],required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--train-size',type=int,default=8);p.add_argument('--dev-size',type=int,default=32)
    p.add_argument('--substructure-epochs',type=int,default=5);p.add_argument('--full-epochs',type=int,default=30)
    p.add_argument('--eval-every',type=int,default=5)
    p.add_argument('--device',default='cuda')
    a=p.parse_args()
    if min(a.train_size,a.dev_size,a.substructure_epochs,a.full_epochs,a.eval_every)<1:p.error('Invalid sizes')
    a.output.mkdir(parents=True,exist_ok=False)
    config=json.loads(Path('configs/text2event/preparation-upstream.json').read_text())
    config.update(constraint_decoder='upstream-compatible',substructure_constraint_decoder='upstream-compatible',
                  constraint_repetition_policy='none',empty_root_penalty=0.)
    folder=a.bundle/a.dataset
    train=fixed_subset(list(map(json.loads,(folder/'train.gold.jsonl').read_text().splitlines())),a.train_size)
    dev=fixed_subset(list(map(json.loads,(folder/'dev.gold.jsonl').read_text().splitlines())),a.dev_size)
    if {r['id'] for r in train}&{r['id'] for r in dev}:raise ValueError('TRAIN/DEV ID overlap')
    write(a.output/'selected-train.json',train);write(a.output/'selected-dev.json',dev)
    write(a.output/'run.json',{'kind':'eight-TRAIN-diagnostic-not-baseline','dataset':a.dataset,'train_size':a.train_size,
        'dev_size':a.dev_size,'substructure_epochs':a.substructure_epochs,'full_epochs':a.full_epochs,
        'seed':421,'lr':1e-4,'batch':1,
        'dev_token_sequences_also_in_selected_train':sum(tuple(r['tokens']) in {tuple(x['tokens']) for x in train} for r in dev),
        'early_stopping':False,'eval_every':a.eval_every,'scheduler':'linear','warmup_steps':0,
        'batch_note':'micro-batch 1 for 8-sample diagnostic; not original effective batch 16',
        'selection':'DEV F1, then DEV teacher loss for ties','config':config})
    random.seed(421);torch.manual_seed(421)
    model,t=load_components(config,a.bundle/'tokenizer');model.to(a.device)
    audit=audit_selected(t,{'train':train,'dev':dev},folder/'event.schema',config)
    write(a.output/'tokenizer-constraint-audit.json',audit)
    if audit['failures']:raise ValueError('Tokenizer/constraint preflight failed')
    model.gradient_checkpointing_enable();model.config.use_cache=False
    write(a.output/'model-loading.json',model.text2event_load_report)
    for stage,converter,epochs in [('substructure',convert_substructure,a.substructure_epochs),('full',convert,a.full_epochs)]:
        if stage=='full':
            model.load_state_dict(torch.load(a.output/'best-substructure.pt',map_location='cpu',weights_only=False)['model'])
            (a.output/'best-substructure.pt').unlink()
        prefix=config['substructure_source_prefix' if stage=='substructure' else 'full_source_prefix']
        train_encoded=encode(train,converter,prefix,t,config);dev_encoded=encode(dev,converter,prefix,t,config)
        loader=DataLoader(train_encoded,batch_size=1,shuffle=True,collate_fn=BatchCollator(t))
        opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=0.)
        scheduler=get_linear_schedule_with_warmup(opt,num_warmup_steps=0,num_training_steps=len(loader)*epochs)
        stopper=None
        best=None
        for epoch in range(1,epochs+1):
            began=time.monotonic();print(f'{stage} {epoch} TRAIN start',flush=True)
            metrics=train_epoch(model,loader,opt,scheduler,a.device,progress_every=16)
            metrics['train_seconds']=time.monotonic()-began
            if epoch%a.eval_every and epoch!=epochs:
                print(json.dumps({'stage':stage,'epoch':epoch,**metrics}),flush=True)
                continue
            disk=save_weights_atomic(a.output/'latest-weights.pt',model,stage=stage,epoch=epoch)
            print(json.dumps({'stage':stage,'epoch':epoch,'EVAL':'starting','disk':disk}),flush=True)
            started=time.monotonic()
            teacher=teacher_probe(model,t,dev_encoded,a.device,max_errors=3)
            write(a.output/f'{stage}-{epoch}-dev-teacher.json',teacher)
            if stage=='substructure':
                scores,issues=evaluate_substructure(model,t,dev,folder/'event.schema',config,a.device)
                f1=scores['f1']
            else:
                scores,pred,issues=evaluate_model(model,t,dev,folder/'event.schema',config,a.device)
                f1=scores['AC']['f1'];metrics['generation']=generation_summary(pred,issues)
                metrics['surface_diagnostics']=surface_diagnostics(dev,issues,t)
                write(a.output/f'full-{epoch}-dev-predictions.json',pred)
                if stopper:stopper.update(f1)
            key=(f1,-teacher['mean_token_loss'])
            if best is None or key>best:
                best=key;save_weights_atomic(a.output/f'best-{stage}.pt',model,stage=stage,epoch=epoch)
                write(a.output/f'best-{stage}.json',{'epoch':epoch,'dev_f1':f1,'teacher_loss':teacher['mean_token_loss']})
            metrics.update(stage=stage,epoch=epoch,dev=scores,teacher_loss=teacher['mean_token_loss'],
                dev_seconds=time.monotonic()-started,early_stopping=stopper.state_dict() if stopper else None)
            write(a.output/f'{stage}-{epoch}.json',metrics);write(a.output/f'{stage}-{epoch}-dev-diagnostics.json',issues)
            print(json.dumps(metrics),flush=True)
            if stopper and stopper.stopped:break
        write(a.output/f'{stage}-completion.json',{'epochs_completed':epoch,'best_key':best,
            'stop_reason':'early_stopping' if stopper and stopper.stopped else 'max_epochs'})
    model.load_state_dict(torch.load(a.output/'best-full.pt',map_location='cpu',weights_only=False)['model'])
    paired={}
    # Compare the same best weights on TRAIN to distinguish memorization from
    # constraint failure. DEV above remains separate and chooses the checkpoint.
    for mode in ['upstream-compatible','unconstrained']:
        current=dict(config,constraint_decoder=mode)
        started=time.monotonic()
        scores,pred,issues=evaluate_model(model,t,train,folder/'event.schema',current,a.device)
        paired[mode]={'scores':scores,'generation':generation_summary(pred,issues),
                      'seconds':time.monotonic()-started}
        write(a.output/f'train-{mode}-diagnostics.json',issues)
        write(a.output/f'train-{mode}-predictions.json',pred)
        write(a.output/'train-decoder-comparison.json',paired)
    write(a.output/'completion.json',{'kind':'eight-TRAIN-diagnostic-not-baseline','dataset':a.dataset,
        'train_size':len(train),'dev_size':len(dev),'test_evaluated':False})


if __name__=='__main__':main()
