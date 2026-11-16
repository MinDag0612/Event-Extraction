"""Resume one GENEVA substructure epoch; persist pre-DEV state, then probe 8 DEV."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import shutil
import time
import torch
from torch.utils.data import DataLoader
from scripts.train_text2event import build_scheduler
from scripts.compare_text2event_checkpoints import decoder_config, flat_limit
from src.baselines.text2event.training import load_components, BundleDataset, BatchCollator, train_epoch
from src.baselines.text2event.checkpoint import restore_checkpoint, save_checkpoint, save_best
from src.baselines.text2event.curriculum import evaluate_substructure
from src.baselines.text2event.evaluation_trace import EvaluationTrace


def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def validate_source(run, bundle):
    if run['dataset']!='GENEVA':raise ValueError('Expected GENEVA checkpoint')
    if not run['gradient_checkpointing']:raise ValueError('Expected fixed gradient-checkpointing recipe')
    for name,digest in run['files'].items():
        with (bundle/name).open('rb') as handle:
            if hashlib.file_digest(handle,'sha256').hexdigest()!=digest:raise ValueError('Bundle differs: '+name)


def validate_full_state(state, expected_epoch):
    if state['signature']['stage']!='substructure' or state['epoch']!=expected_epoch:
        raise ValueError('Unexpected full-state stage/epoch')
    if not {'model','optimizer','scheduler','python_rng','torch_rng','cuda_rng'} <= set(state):
        raise ValueError('Full optimizer/scheduler/RNG state required')
    return state['signature']['run']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True)
    sources=p.add_mutually_exclusive_group(required=True)
    sources.add_argument('--source',type=Path,help='Pinned baseline-GENEVA run directory')
    sources.add_argument('--state-checkpoint',type=Path,help='Trusted full-state pre-DEV snapshot')
    p.add_argument('--expected-source-epoch',type=int,default=1)
    p.add_argument('--disallow-empty-tree',action='store_true')
    p.add_argument('--reserve-free-gib',type=float,default=12)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('GPU required; no training started')
    args.output.mkdir(parents=True,exist_ok=False)
    trace=EvaluationTrace(args.output/'trace.jsonl')
    if args.state_checkpoint:
        metadata=torch.load(args.state_checkpoint,map_location='cpu',mmap=True,weights_only=False)
        run=validate_full_state(metadata,args.expected_source_epoch);del metadata
        source=args.state_checkpoint
    else:
        run=json.loads((args.source/'run.json').read_text())
        source=args.source/'substructure/last.pt'
    validate_source(run,args.bundle)
    target_epoch=args.expected_source_epoch+1
    if not 1 <= args.expected_source_epoch < run['substructure_epochs']:
        raise ValueError('No remaining substructure epoch in source recipe')
    config=run['config'];folder=args.bundle/'GENEVA'
    if not source.is_file():raise FileNotFoundError('Full optimizer/RNG checkpoint is required')
    write(args.output/'request.json',{'kind':'one-resumed-substructure-epoch-then-8DEV-NOT-baseline',
        'source':str(source),'source_run':run,'expected_completed_epochs':args.expected_source_epoch,'target_epoch':target_epoch,
        'dev_samples':8,'test_used':False,'full_stage_started':False,
        'wall_clock_timeout':None,'source_copied':False})
    if args.reserve_free_gib<10:raise ValueError('Snapshot reserve must be at least10GiB')
    if shutil.disk_usage(args.output).free<args.reserve_free_gib*1024**3:
        raise OSError(f'Need {args.reserve_free_gib} GiB free for pre-DEV state and weights')
    random.seed(run['seed']);torch.manual_seed(run['seed'])
    trace.emit('MODEL_LOAD_START')
    model,tokenizer=load_components(config,args.bundle/'tokenizer');model.to('cuda')
    model.gradient_checkpointing_enable();model.config.use_cache=False
    ds=BundleDataset(folder,'train',tokenizer,config['max_source_length'],config['max_target_length'],
                     'substructure',source_prefix=config.get('substructure_source_prefix','span: '))
    loader=DataLoader(ds,batch_size=run['micro_batch'],shuffle=True,collate_fn=BatchCollator(tokenizer))
    optimizer=torch.optim.AdamW(model.parameters(),lr=run['lr'],weight_decay=0.)
    scheduler,schedule=build_scheduler(optimizer,len(loader),run['accumulation'],run['substructure_epochs'],
         run['scheduler'],run['warmup_steps'],run.get('warmup_ratio'))
    signature={'run':run,'stage':'substructure'}
    trace.emit('RESTORE_START')
    epoch,best=restore_checkpoint(source,model,optimizer,scheduler,signature)
    if epoch!=args.expected_source_epoch:raise ValueError('Unexpected source epoch; training cancelled')
    expected_updates=(schedule['total_optimizer_updates']//run['substructure_epochs'])*epoch
    if scheduler.last_epoch!=expected_updates:raise ValueError('Unexpected source scheduler update')
    trace.emit('RESTORED',completed_epochs=epoch,scheduler_update=scheduler.last_epoch)
    trace.emit('TRAIN_START',stage='substructure',epoch=target_epoch,batches=len(loader))
    began=time.perf_counter()
    metrics=train_epoch(model,loader,optimizer,scheduler,'cuda',run['accumulation'],progress_every=64)
    metrics['train_seconds']=time.perf_counter()-began
    write(args.output/f'train-epoch-{target_epoch}.json',metrics)
    trace.emit('TRAIN_END',epoch=target_epoch,**metrics)
    # Pre-DEV snapshots are separate artifacts, never baseline last/best.pt.
    trace.emit('SNAPSHOT_START',epoch=target_epoch,dev_evaluated=False)
    save_checkpoint(args.output/'pre-dev-state.pt',model,optimizer,scheduler,target_epoch,best,signature)
    save_best(args.output/'pre-dev-weights.pt',model,target_epoch,best,signature)
    write(args.output/'snapshot.json',{'epoch':target_epoch,'phase':'before_DEV','dev_evaluated':False,
          'previous_best_dev':best,'state':'pre-dev-state.pt','weights':'pre-dev-weights.pt',
          'scheduler_update':scheduler.last_epoch})
    trace.emit('SNAPSHOT_SAVED',epoch=target_epoch,scheduler_update=scheduler.last_epoch)
    # The optimizer is no longer used; release its GPU state before generation.
    del optimizer,scheduler,loader,ds
    torch.cuda.empty_cache()
    evaluation=decoder_config(copy.deepcopy(config),'project-constrained')
    evaluation['constraint_allow_empty_tree']=not args.disallow_empty_tree
    train=list(map(json.loads,(folder/'train.gold.jsonl').read_text().splitlines()))
    evaluation['constraint_repetition_limits']['GENEVA']['flat_repeat_limit']=flat_limit(train)
    dev=list(map(json.loads,(folder/'dev.gold.jsonl').read_text().splitlines()))
    indices=sorted(random.Random(421).sample(range(len(dev)),8));samples=[dev[i] for i in indices]
    write(args.output/'evaluation-request.json',{'ids':[x['id'] for x in samples],
          'selection_seed':421,'config':evaluation,'samples':8,'checkpoint_epoch':target_epoch})
    trace.emit('DEV_START',epoch=target_epoch,samples=8)
    began=time.perf_counter()
    scores,issues=evaluate_substructure(model,tokenizer,samples,folder/'event.schema',evaluation,'cuda',observer=trace)
    summary={'epoch':target_epoch,'samples':8,'scores':scores,'dev_seconds':time.perf_counter()-began,
       'invalid':sum(x['status']!='ok' for x in issues),
       'true_empty':sum(x['status']=='ok' and x.get('generated','').replace(' ','')=='<extra_id_0><extra_id_1>' for x in issues),
       'timing_totals':{name:sum(x['timing'].get(name,0) for x in issues)
          for name in ['encode','constraint','generate','callback_seconds','sample_seconds']}}
    write(args.output/'diagnostics.json',issues);write(args.output/'summary.json',summary)
    write(args.output/'completion.json',{'additional_training_epochs':1,'substructure_epoch':target_epoch,
          'dev_samples':8,'full_DEV_completed':False,'full_stage_started':False,'test_used':False})
    trace.emit('DIAGNOSTIC_COMPLETE',**summary)


if __name__=='__main__':main()
