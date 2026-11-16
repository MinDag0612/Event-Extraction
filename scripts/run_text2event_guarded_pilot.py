"""Run one frozen-adaptation substructure epoch with recoverable DEV evaluation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time


def write(path,value):
    """Commit an artifact atomically."""
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temporary.replace(path)


def digest(path):
    """Hash a source or checkpoint without reading it all at once."""
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def stalled(event,now,sample_limit,progress_limit,io_limit):
    """Choose a deadline by phase without imposing a total run duration."""
    phase=event['phase']
    limit=sample_limit if phase=='SAMPLE_START' else (progress_limit if phase=='TRAIN_PROGRESS' else io_limit)
    return now-event['time']>limit


def counts_score(tp,fp,fn):
    """Aggregate the original flat leaf-set counts across completed examples."""
    return {'precision':tp/(tp+fp) if tp+fp else 0.,'recall':tp/(tp+fn) if tp+fn else 0.,
            'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.}


def worker(a):
    """Train using frozen components and persist state before any DEV generation."""
    output=a.output;output.mkdir(parents=True,exist_ok=True)
    def event(phase,**fields):
        value={'phase':phase,'time':time.time(),**fields};write(output/'activity.json',value);print(json.dumps(value),flush=True)
    event('PREFLIGHT')
    lock=json.loads(a.lock.read_text());source=a.source.resolve()
    if digest(source/'package-manifest.json')!=lock['source_manifest_sha256']:raise ValueError('Unexpected VHE source manifest')
    for name,h in json.loads((source/'package-manifest.json').read_text()).items():
        if digest(source/name)!=h:raise ValueError('Frozen source mismatch: '+name)
    sys.path.insert(0,str(source))
    import random
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer
    from scripts.train_text2event import build_scheduler
    from src.baselines.text2event.training import load_components,BundleDataset,BatchCollator,train_epoch
    from src.baselines.text2event.checkpoint import save_checkpoint,save_best
    from src.baselines.text2event.curriculum import evaluate_substructure,leaf_set
    from src.baselines.text2event.convert import convert_substructure
    config=json.loads((source/'configs/text2event/baseline-mt5.json').read_text())
    if config!=lock['config']:raise ValueError('Configuration differs from VHE')
    folder=source/'bundle/PHEE';tokenizer=AutoTokenizer.from_pretrained(source/'bundle/tokenizer',local_files_only=True)
    preflight={}
    for stage,prefix in [('substructure',config['substructure_source_prefix']),('full',config['full_source_prefix'])]:
        for split in ['train','dev']:
            ds=BundleDataset(folder,split,tokenizer,config['max_source_length'],config['max_target_length'],stage,prefix)
            preflight[stage+'-'+split]={'samples':len(ds),'max_source':max(len(r['input_ids']) for r in ds.examples),
                                      'max_target':max(len(r['labels']) for r in ds.examples)}
    if preflight['full-train']['samples']!=2853 or preflight['full-dev']['samples']!=949:raise ValueError('Unexpected PHEE split size')
    write(output/'preflight.json',preflight)
    signature={'implementation':lock['implementation'],'config':config,'dataset':'PHEE',**lock['trainer'],
        'files':{str(p.relative_to(source/'bundle')):digest(p) for p in list(folder.glob('*'))+list((source/'bundle/tokenizer').glob('*')) if p.is_file()}}
    write(output/'run.json',signature)
    write(output/'guard-config.json',{'sample_stall_seconds':a.sample_stall_seconds,'train_progress_stall_seconds':a.progress_stall_seconds,
        'io_stall_seconds':a.io_stall_seconds,'consecutive_without_eos':a.consecutive_without_eos,
        'total_wall_clock_limit':None,'test_evaluated':False,'first_invocation_epochs':1})
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    event('MODEL_LOAD');random.seed(signature['seed']);torch.manual_seed(signature['seed'])
    model,tokenizer=load_components(config,source/'bundle/tokenizer');model.to('cuda')
    model.gradient_checkpointing_enable();model.config.use_cache=False
    random.seed(signature['seed']);torch.manual_seed(signature['seed'])
    ds=BundleDataset(folder,'train',tokenizer,config['max_source_length'],config['max_target_length'],'substructure',config['substructure_source_prefix'])
    loader=DataLoader(ds,batch_size=signature['micro_batch'],shuffle=True,collate_fn=BatchCollator(tokenizer))
    optimizer=torch.optim.AdamW(model.parameters(),lr=signature['lr'],weight_decay=0.)
    scheduler,schedule=build_scheduler(optimizer,len(loader),signature['accumulation'],signature['substructure_epochs'],signature['scheduler'],signature['warmup_steps'],signature['warmup_ratio'])
    write(output/'substructure/schedule.json',schedule)
    class Observer:
        def before_step(self,model,optimizer,gradient_norm):pass
        def after_step(self,model):event('TRAIN_PROGRESS',optimizer_update=scheduler.last_epoch)
    event('TRAIN_PROGRESS',optimizer_update=0);started=time.monotonic()
    metrics=train_epoch(model,loader,optimizer,scheduler,'cuda',signature['accumulation'],progress_every=64,step_observer=Observer())
    metrics['train_seconds']=time.monotonic()-started;write(output/'train-epoch-1.json',metrics)
    stage_signature={'run':signature,'stage':'substructure'};snapshot=output/'substructure/pre-dev-state.pt'
    event('CHECKPOINT_SAVE')
    if __import__('shutil').disk_usage(output).free<10*1024**3:raise OSError('Insufficient space before pre-DEV checkpoint')
    save_checkpoint(snapshot,model,optimizer,scheduler,1,-1.,stage_signature)
    write(output/'snapshot.json',{'stage':'substructure','epoch':1,'phase':'before_DEV','dev_evaluated':False,
        'scheduler_update':scheduler.last_epoch,'checkpoint':'substructure/pre-dev-state.pt','checkpoint_sha256':digest(snapshot),
        'best_is_previous_dev_score':True,'resume_action':'restore snapshot then finish DEV before continuing epoch2'})
    event('DEV_READY');golds=list(map(json.loads,(folder/'dev.gold.jsonl').read_text().splitlines()))
    tp=fp=fn=0;issues=[];streak=0;started=time.monotonic();normalize=tokenizer.backend_tokenizer.normalizer.normalize_str
    with (output/'dev-samples.jsonl').open('w') as log:
        for i,sample in enumerate(golds,1):
            event('SAMPLE_START',index=i,total=len(golds),id=sample['id'])
            _,one=evaluate_substructure(model,tokenizer,[sample],folder/'event.schema',config,'cuda');issue=one[0];issues.append(issue)
            predicted=leaf_set(issue['generated'],normalize) if issue['status']=='ok' else set()
            gold=leaf_set(convert_substructure(sample)['event'],normalize)
            tp+=len(predicted&gold);fp+=len(predicted-gold);fn+=len(gold-predicted)
            log.write(json.dumps(issue,ensure_ascii=False)+'\n');log.flush()
            streak=streak+1 if issue.get('reason')=='generation_limit_without_eos' else 0
            event('SAMPLE_DONE',index=i,total=len(golds),status=issue['status'],consecutive_without_eos=streak)
            if streak>=a.consecutive_without_eos:
                write(output/'status.json',{'status':'evaluation_interrupted','reason':'consecutive_generation_limit_without_eos',
                    'evaluated_samples':i,'total_dev_samples':len(golds),'baseline_complete':False,'dev_complete':False,
                    'resume_required':True,'checkpoint':'substructure/pre-dev-state.pt','test_evaluated':False})
                return
    score=counts_score(tp,fp,fn);metrics.update(epoch=1,stage='substructure',dev=score,dev_seconds=time.monotonic()-started,
        early_stopping=None,selection_metric='flat-label-surface-F1')
    write(output/'substructure/epoch-1-metrics.json',metrics);write(output/'substructure/epoch-1-diagnostics.json',issues)
    event('CHECKPOINT_SAVE')
    if __import__('shutil').disk_usage(output).free<3*1024**3:raise OSError('Insufficient space for selected model')
    save_best(output/'substructure/best.pt',model,1,score['f1'],stage_signature)
    write(output/'epoch-1-dev-commit.json',{'dev_complete':True,'epoch':1,'stage':'substructure',
        'dev_samples':len(golds),'best':score['f1'],'checkpoint':'substructure/pre-dev-state.pt',
        'resume_action':'restore pre-dev-state, apply this completed DEV best score, then continue epoch2'})
    write(output/'invocation.json',{'stop_reason':'invocation_epoch_budget','stage':'substructure','stage_epoch':1,
        'resume_required':True,'resume_protocol':'pre-dev-state plus epoch-1-dev-commit.json; not stock --resume'})
    write(output/'status.json',{'status':'invocation_complete','returncode':0,'dev_complete':True,'dev_samples':len(golds),
        'resume_required':True,'baseline_complete':False,'test_evaluated':False,'dev':score,
        'checkpoint':'substructure/pre-dev-state.pt','dev_commit':'epoch-1-dev-commit.json'})


def supervise(a,worker_script=None):
    """Stop only stalled phases or an explicitly reported generation failure."""
    a.output.mkdir(parents=True,exist_ok=False)
    command=[sys.executable,'-u',str(Path(worker_script or __file__).resolve()),*sys.argv[1:],'--worker']
    with (a.output/'pilot.log').open('w') as log:
        child=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,start_new_session=True)
        def pump():
            for line in child.stdout:print(line,end='',flush=True);log.write(line);log.flush()
        thread=threading.Thread(target=pump,daemon=True);thread.start()
        began=time.time()
        while child.poll() is None:
            activity=a.output/'activity.json'
            event=json.loads(activity.read_text()) if activity.exists() else {'phase':'BOOTSTRAP','time':began}
            if stalled(event,time.time(),a.sample_stall_seconds,a.progress_stall_seconds,a.io_stall_seconds):
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                snapshot=a.output/'substructure/pre-dev-state.pt'
                write(a.output/'status.json',{'status':'evaluation_interrupted' if event['phase']=='SAMPLE_START' else 'run_interrupted',
                    'reason':'phase_progress_watchdog','last_activity':event,'snapshot_present':snapshot.exists(),
                    'resume_required':True,'baseline_complete':False,'dev_complete':False,'test_evaluated':False})
                break
            time.sleep(1)
        thread.join(timeout=5)
        if not (a.output/'status.json').exists():
            write(a.output/'status.json',{'status':'failed','returncode':child.returncode,'baseline_complete':False,
                'resume_required':True,'test_evaluated':False})
    print('PILOT_STATUS',(a.output/'status.json').read_text(),flush=True)
    status=json.loads((a.output/'status.json').read_text())
    if status['status']=='failed':raise RuntimeError('Pilot worker failed; inspect pilot.log')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['source','lock','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--sample-stall-seconds',type=int,default=120)
    p.add_argument('--progress-stall-seconds',type=int,default=300)
    p.add_argument('--io-stall-seconds',type=int,default=900)
    p.add_argument('--consecutive-without-eos',type=int,default=8)
    p.add_argument('--worker',action='store_true');a=p.parse_args()
    if min(a.sample_stall_seconds,a.progress_stall_seconds,a.io_stall_seconds,a.consecutive_without_eos)<1:p.error('Guard limits must be positive')
    if a.worker:worker(a)
    else:supervise(a)
