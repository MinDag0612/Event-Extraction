"""Text2Event-mT5: upstream-style flat substructure pretraining, then full trees.

--epochs counts FULL-stage epochs; --substructure-epochs is additional.
Separate optimizer/scheduler per stage; best substructure weights initialize full.
DEV evaluation/checkpoint selection is epoch-boundary (project adaptation).
"""
import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import get_constant_schedule_with_warmup, get_linear_schedule_with_warmup
from src.baselines.text2event.training import load_components, BundleDataset, BatchCollator, train_epoch
from src.baselines.text2event.evaluation import evaluate_model
from src.baselines.text2event.curriculum import evaluate_substructure
from src.baselines.text2event.checkpoint import save_checkpoint, restore_checkpoint, save_best, same_training_except_full_stopping
from src.baselines.text2event.diagnostics import generation_summary
from src.baselines.text2event.early_stopping import EarlyStopping
from src.baselines.text2event.evaluation_trace import EvaluationTrace


def build_scheduler(optimizer, loader_length, accumulation, epochs, kind, warmup_steps=0, warmup_ratio=None):
    total=math.ceil(loader_length/accumulation)*epochs
    warmup=math.ceil(total*warmup_ratio) if warmup_ratio is not None else warmup_steps
    if kind=='linear':
        scheduler=get_linear_schedule_with_warmup(optimizer,num_warmup_steps=warmup,num_training_steps=total)
    else:
        scheduler=get_constant_schedule_with_warmup(optimizer,num_warmup_steps=warmup)
    return scheduler,{'kind':kind,'total_optimizer_updates':total,'warmup_updates':warmup}


def prune_completed_stage(output):
    """Remove obsolete optimizer state only after stage completion; keep best weights."""
    if not (output/'completion.json').exists() or not (output/'best.pt').exists():
        raise ValueError('Cannot prune stage without completion marker and best weights')
    last=output/'last.pt'
    if last.exists():
        size=last.stat().st_size;last.unlink()
        print(json.dumps({'pruned_completed_stage':str(output),'released_bytes':size}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--dataset',choices=['BKEE','GENEVA','PHEE','VHE'],required=True)
    p.add_argument('--config',type=Path,default=Path('configs/text2event/preparation.json'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=30,help='FULL stage epochs, in addition to substructure')
    p.add_argument('--substructure-epochs',type=int,default=5)
    p.add_argument('--micro-batch',type=int,default=1)
    p.add_argument('--accumulation',type=int,default=16)
    p.add_argument('--lr',type=float,default=5e-5)
    p.add_argument('--warmup-steps',type=int,default=0)
    p.add_argument('--lr-scheduler',choices=['constant_with_warmup','linear'],default='constant_with_warmup')
    p.add_argument('--warmup-ratio',type=float)
    p.add_argument('--seed',type=int,default=421)
    p.add_argument('--patience',type=int,default=5,help='Full-stage patience; 0 disables early stopping')
    p.add_argument('--min-delta',type=float,default=0.001)
    p.add_argument('--min-full-epochs',type=int,default=5,help='No patience counted during these full epochs')
    p.add_argument('--max-epochs-this-run',type=int,help='Invocation budget; stop at epoch boundary, resume with same signature')
    p.add_argument('--gradient-checkpointing',action='store_true')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--final-evaluation',action='store_true')
    p.add_argument('--prune-completed-state',action='store_true',help='Remove completed substructure optimizer state; keep best weights and metrics')
    p.add_argument('--progress-every',type=int,default=0,help='Print training progress every N microbatches')
    p.add_argument('--evaluation-progress',action='store_true',help='Persist per-sample DEV phases and constraint callback timings')
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    a=p.parse_args()
    if a.progress_every<0:p.error('Progress interval must be nonnegative')
    if a.warmup_ratio is not None and (not 0<=a.warmup_ratio<1 or a.warmup_steps):
        p.error('Warmup ratio must be in [0,1); do not combine with warmup steps')
    if min(a.epochs,a.substructure_epochs,a.micro_batch,a.accumulation)<1 or a.lr<=0 or min(a.warmup_steps,a.patience,a.min_full_epochs)<0 or (a.max_epochs_this_run is not None and a.max_epochs_this_run<1):
        p.error('Require positive epochs/batches/lr and nonnegative warmup/patience')
    if a.output.exists() and not a.resume:
        raise FileExistsError(a.output)
    if a.resume and not (a.output/'run.json').exists():
        raise FileNotFoundError('Resume requires a new-curriculum run.json')
    config=json.loads(a.config.read_text());folder=a.bundle/a.dataset
    signature={'implementation':'mt5-upstream-curriculum-es-v2','config':config,
        'dataset':a.dataset, 'full_epochs':a.epochs,'substructure_epochs':a.substructure_epochs,
        'micro_batch':a.micro_batch,'accumulation':a.accumulation,'lr':a.lr,
        'scheduler':a.lr_scheduler,'warmup_steps':a.warmup_steps,'seed':a.seed,
        'patience':a.patience,'min_delta':a.min_delta,'min_full_epochs':a.min_full_epochs,'gradient_checkpointing':a.gradient_checkpointing,
        'files':{str(f.relative_to(a.bundle)):hashlib.sha256(f.read_bytes()).hexdigest()
                 for f in list(folder.glob('*'))+list((a.bundle/'tokenizer').glob('*')) if f.is_file()}}
    if a.warmup_ratio is not None:signature['warmup_ratio']=a.warmup_ratio
    if a.resume:
        previous=json.loads((a.output/'run.json').read_text())
        if previous!=signature:
            full_started=(a.output/'full').exists() and any((a.output/'full').iterdir())
            if full_started or not same_training_except_full_stopping(previous,signature):
                raise ValueError('Resume signature differs; only future full early stopping may change before full starts')
            print(json.dumps({'resume_change':'future full early stopping',
                              'patience':a.patience,'min_delta':a.min_delta,
                              'min_full_epochs':a.min_full_epochs}),flush=True)
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'run.json').write_text(json.dumps(signature,indent=2)+'\n')
    random.seed(a.seed);torch.manual_seed(a.seed)
    model,tokenizer=load_components(config,a.bundle/'tokenizer');model.to(a.device)
    (a.output/'model-loading.json').write_text(json.dumps(model.text2event_load_report,indent=2)+'\n')
    if a.gradient_checkpointing:
        model.gradient_checkpointing_enable();model.config.use_cache=False
    dev=list(map(json.loads,(folder/'dev.gold.jsonl').read_text().splitlines()))
    invocation_epochs=0
    for stage,epochs,prefix in [('substructure',a.substructure_epochs,config.get('substructure_source_prefix','span: ')),
                                ('full',a.epochs,config.get('full_source_prefix','event: '))]:
        output=a.output/stage;output.mkdir(exist_ok=True)
        if stage=='full':
            state=torch.load(a.output/'substructure'/'best.pt',map_location='cpu',weights_only=False)
            model.load_state_dict(state['model'])
        if (output/'completion.json').exists():
            # Stage completion is atomic and always follows the final checkpoint.
            if a.prune_completed_state and stage=='substructure':prune_completed_stage(output)
            continue
        # Match two independent training invocations: transfer weights only,
        # reset optimization and seed; resume within a stage restores full state.
        random.seed(a.seed);torch.manual_seed(a.seed)
        ds=BundleDataset(folder,'train',tokenizer,config['max_source_length'],config['max_target_length'],stage,source_prefix=prefix)
        loader=DataLoader(ds,batch_size=a.micro_batch,shuffle=True,collate_fn=BatchCollator(tokenizer))
        optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=0.)
        scheduler,schedule=build_scheduler(optimizer,len(loader),a.accumulation,epochs,a.lr_scheduler,a.warmup_steps,a.warmup_ratio)
        (output/'schedule.json').write_text(json.dumps(schedule,indent=2)+'\n')
        stopper=EarlyStopping(a.patience,a.min_delta,min_epochs=a.min_full_epochs) if stage=='full' and a.patience else None
        stage_signature={'run':signature,'stage':stage}
        start=0;best=-1.;completed=0
        if a.resume and (output/'last.pt').exists():
            start,best=restore_checkpoint(output/'last.pt',model,optimizer,scheduler,stage_signature,early_stopping=stopper)
            completed=start
            print(json.dumps({'resumed_stage':stage,'completed_epochs':start,'scheduler_update':scheduler.last_epoch}),flush=True)
        for epoch in range(start,epochs):
            if stopper and stopper.stopped:break
            began=time.monotonic()
            print(json.dumps({'stage':stage,'epoch':epoch+1,'phase':'TRAIN_START'}),flush=True)
            extra={'progress_every':a.progress_every} if a.progress_every else {}
            metrics=train_epoch(model,loader,optimizer,scheduler,a.device,a.accumulation,**extra)
            train_seconds=time.monotonic()-began
            print(json.dumps({'stage':stage,'epoch':epoch+1,'phase':'DEV_START','train_seconds':train_seconds}),flush=True)
            evaluation_started=time.monotonic()
            trace={'observer':EvaluationTrace(output/f'epoch-{epoch+1}-trace.jsonl')} if a.evaluation_progress else {}
            if stage=='substructure':
                scores,issues=evaluate_substructure(model,tokenizer,dev,folder/'event.schema',config,a.device,**trace)
                score=scores['f1'];pred=None
            else:
                scores,pred,issues=evaluate_model(model,tokenizer,dev,folder/'event.schema',config,a.device,**trace)
                score=scores['AC']['f1'];metrics['generation']=generation_summary(pred,issues)
            if stopper:stopper.update(score)
            metrics.update(epoch=epoch+1,stage=stage,dev=scores,
                           train_seconds=train_seconds, dev_seconds=time.monotonic()-evaluation_started,
                           early_stopping=stopper.state_dict() if stopper else None,
                           selection_metric='flat-label-surface-F1' if stage=='substructure' else 'DEV AC F1')
            if score>best:
                best=score;save_best(output/'best.pt',model,epoch+1,best,stage_signature)
            save_checkpoint(output/'last.pt',model,optimizer,scheduler,epoch+1,best,stage_signature,early_stopping=stopper)
            for suffix,value in [('metrics',metrics),('diagnostics',issues),('predictions',pred)]:
                if value is not None:(output/f'epoch-{epoch+1}-{suffix}.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
            completed=epoch+1;invocation_epochs+=1;print(json.dumps(metrics),flush=True)
            if a.max_epochs_this_run and invocation_epochs>=a.max_epochs_this_run and not (stopper and stopper.stopped) and (completed<epochs or stage=='substructure'):
                (a.output/'invocation.json').write_text(json.dumps({'stop_reason':'invocation_epoch_budget','stage':stage,'stage_epoch':completed,'resume_required':True})+'\n')
                return
        completion={'stage':stage,'epochs_completed':completed,'best_dev_f1':best,
                    'early_stopping':stopper.state_dict() if stopper else None,
                    'stop_reason':'early_stopping' if stopper and stopper.stopped else 'max_epochs'}
        temporary=output/'completion.tmp';temporary.write_text(json.dumps(completion,indent=2)+'\n');temporary.replace(output/'completion.json')
        if a.prune_completed_state and stage=='substructure':prune_completed_stage(output)
    state=torch.load(a.output/'full'/'best.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(state['model'])
    if a.final_evaluation:
        for split in ['train','test']:
            gold=list(map(json.loads,(folder/f'{split}.gold.jsonl').read_text().splitlines()))
            scores,pred,issues=evaluate_model(model,tokenizer,gold,folder/'event.schema',config,a.device)
            (a.output/f'{split}-results.json').write_text(json.dumps({'checkpoint_epoch':state['epoch'],'scores':scores,'diagnostics':issues},ensure_ascii=False,indent=2)+'\n')
            (a.output/f'{split}-predictions.json').write_text(json.dumps(pred,ensure_ascii=False)+'\n')
        model.save_pretrained(a.output/'best-model');tokenizer.save_pretrained(a.output/'best-model')
    (a.output/'invocation.json').unlink(missing_ok=True)
    (a.output/'completion.json').write_text(json.dumps({'substructure':json.loads((a.output/'substructure/completion.json').read_text()),
        'full':json.loads((a.output/'full/completion.json').read_text()),'best_full_epoch':state['epoch']},indent=2)+'\n')


if __name__=='__main__':main()
