"""Resume paired epoch2 branches with identical state and different LR schedules."""
import hashlib
import json
import pickle
import random
import shutil
import time
import torch
from torch.utils.data import DataLoader
from scripts.train_text2event import build_scheduler
from src.baselines.text2event.training import load_components, BundleDataset, BatchCollator, train_epoch
from src.baselines.text2event.checkpoint import restore_checkpoint, save_checkpoint, save_best
from src.baselines.text2event.curriculum import evaluate_substructure
from src.baselines.text2event.evaluation_trace import EvaluationTrace
from src.baselines.text2event_clean.preflight import bundle_hashes
from src.baselines.text2event_clean.checkpoint_audit import validate_checkpoint
from src.baselines.text2event_clean.pilot import PilotObserver


def configure_branch_lr(optimizer, scheduler, branch):
    """Preserve restored optimizer moments; change only the post-warmup LR schedule."""
    if branch=='linear':return scheduler
    if branch!='constant':raise ValueError('Unknown LR branch')
    return torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:1.,last_epoch=scheduler.last_epoch-1)


class OrderCollator:
    """Record actual microbatch IDs while delegating tensor collation unchanged."""
    def __init__(self,tokenizer):self.base=BatchCollator(tokenizer);self.ids=[]
    def __call__(self,rows):
        self.ids.extend(row['id'] for row in rows)
        return self.base(rows)


def rng_digest():
    """Fingerprint restored Python, CPU and CUDA RNG states."""
    h=hashlib.sha256(pickle.dumps(random.getstate()))
    h.update(torch.get_rng_state().numpy().tobytes())
    for state in torch.cuda.get_rng_state_all():h.update(state.cpu().numpy().tobytes())
    return h.hexdigest()


def run(bundle,config,recipe,source,output,branch,consume_source=False):
    """Restore epoch1, train epoch2, save before evaluation and score fixed TRAIN/DEV."""
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    output.mkdir(parents=True,exist_ok=False)
    def save(name,value):(output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2))
    metadata=torch.load(source,map_location='cpu',mmap=True,weights_only=False)
    validate_checkpoint(metadata,config,bundle,1)
    if any(metadata['signature']['run'].get(k)!=v for k,v in recipe.items()):
        raise ValueError('Source training recipe differs')
    if not {'optimizer','scheduler','python_rng','torch_rng','cuda_rng'}<=set(metadata):
        raise ValueError('Full-state source required')
    source_signature=metadata['signature'];del metadata
    trace=EvaluationTrace(output/'trace.jsonl')
    save('request.json',{'branch':branch,'source_epoch':1,'target_epoch':2,'source_signature':source_signature,
        'only_training_change':'post-warmup LR schedule','config':config,'recipe':recipe,
        'consume_downloaded_source_after_restore':consume_source,'test_used':False})
    trace.emit('MODEL_LOAD_START',branch=branch)
    model,t=load_components(config,bundle/'tokenizer');model.to('cuda')
    model.gradient_checkpointing_enable();model.config.use_cache=False
    folder=bundle/'GENEVA'
    ds=BundleDataset(folder,'train',t,config['max_source_length'],config['max_target_length'],
                     'substructure',config['substructure_source_prefix'])
    collator=OrderCollator(t)
    loader=DataLoader(ds,batch_size=recipe['micro_batch'],shuffle=True,collate_fn=collator)
    optimizer=torch.optim.AdamW(model.parameters(),lr=recipe['lr'],weight_decay=0.)
    scheduler,schedule=build_scheduler(optimizer,len(loader),recipe['accumulation'],recipe['substructure_epochs'],
                                      recipe['scheduler'],warmup_ratio=recipe['warmup_ratio'])
    epoch,best=restore_checkpoint(source,model,optimizer,scheduler,source_signature)
    if epoch!=1 or scheduler.last_epoch!=118:raise ValueError('Expected epoch1/update118 source')
    before_rng=rng_digest()
    scheduler=configure_branch_lr(optimizer,scheduler,branch)
    if rng_digest()!=before_rng:raise ValueError('LR override changed RNG')
    save('restore.json',{'epoch':epoch,'scheduler_update':scheduler.last_epoch,'rng_sha256':before_rng,
        'lr_before_first_step':scheduler.get_last_lr(),'optimizer_state_tensors':len(optimizer.state),
        'source_signature_sha256':hashlib.sha256(json.dumps(source_signature,sort_keys=True).encode()).hexdigest()})
    if consume_source:
        if source.parent.resolve()!=output.parent.resolve() or source.name!='source-epoch1.pt':
            raise ValueError('Refusing to remove a non-temporary source')
        source.unlink()
    if shutil.disk_usage(output).free<10*1024**3:raise OSError('Need10GiB for new snapshots')
    trace.emit('RESTORED',branch=branch,epoch=1,scheduler_update=118,lr=scheduler.get_last_lr())
    indices=sorted(random.Random(421).sample(range(len(ds)),8))
    observer=PilotObserver(output/'optimizer-updates.jsonl',t,[ds[i] for i in indices],'cuda',{1,8,30,64,118})
    observer.epoch=2;observer.measure(model)
    trace.emit('TRAIN_START',branch=branch,epoch=2,samples=len(ds))
    started=time.perf_counter()
    metrics=train_epoch(model,loader,optimizer,scheduler,'cuda',recipe['accumulation'],progress_every=64,step_observer=observer)
    metrics['seconds']=time.perf_counter()-started;save('train.json',metrics)
    save('training-order.json',{'ids':collator.ids,'sha256':hashlib.sha256('\n'.join(collator.ids).encode()).hexdigest()})
    run_signature={**source_signature['run'],'implementation':'text2event-clean-lr-ablation-v1',
        'scheduler':'linear' if branch=='linear' else 'constant_after_epoch1','ablation_branch':branch,
        'source_signature_sha256':hashlib.sha256(json.dumps(source_signature,sort_keys=True).encode()).hexdigest()}
    signature={'run':run_signature,'stage':'substructure'}
    save_checkpoint(output/'pre-dev-state.pt',model,optimizer,scheduler,2,best,signature)
    save_best(output/'pre-dev-weights.pt',model,2,best,signature)
    save('snapshot.json',{'epoch':2,'phase':'before_evaluation','scheduler_update':scheduler.last_epoch,
         'branch':branch,'state':'pre-dev-state.pt','weights':'pre-dev-weights.pt'})
    trace.emit('SNAPSHOT_SAVED',branch=branch,epoch=2,scheduler_update=scheduler.last_epoch)
    del optimizer,scheduler,loader,ds
    torch.cuda.empty_cache()
    for split in ['train','dev']:
        rows=list(map(json.loads,(folder/f'{split}.gold.jsonl').read_text().splitlines()))
        indices=sorted(random.Random(421).sample(range(len(rows)),8));samples=[rows[i] for i in indices]
        trace.emit('EVALUATION_START',branch=branch,split=split,samples=8)
        scores,issues=evaluate_substructure(model,t,samples,folder/'event.schema',config,'cuda',observer=trace)
        save(f'{split}-diagnostics.json',issues)
        save(f'{split}-summary.json',{'ids':[x['id'] for x in samples],'scores':scores,'samples':8,
             'invalid':sum(x['status']!='ok' for x in issues),
             'true_empty':sum(x['status']=='ok' and x.get('generated','').replace(' ','')=='<extra_id_0><extra_id_1>' for x in issues)})
    save('completion.json',{'branch':branch,'epoch':2,'optimizer_updates':metrics['optimizer_updates'],
        'train_samples':len(collator.ids),'train_eval_samples':8,'dev_eval_samples':8,'test_used':False})
    trace.emit('BRANCH_COMPLETE',branch=branch)
