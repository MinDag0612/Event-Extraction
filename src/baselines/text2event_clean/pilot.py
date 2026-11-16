"""Train one fresh substructure epoch and measure the fixed recipe update path."""
import json
import random
import shutil
import time
import torch
from torch.utils.data import DataLoader
from scripts.train_text2event import build_scheduler
from src.baselines.text2event.training import load_components, BundleDataset, BatchCollator, train_epoch
from src.baselines.text2event.checkpoint import save_checkpoint, save_best
from src.baselines.text2event.curriculum import evaluate_substructure
from src.baselines.text2event.evaluation_trace import EvaluationTrace
from src.baselines.text2event.update_probe import UpdateProbe
from src.baselines.text2event_clean.preflight import bundle_hashes


class PilotObserver(UpdateProbe):
    """Record every update and fixed TRAIN loss without changing training RNG."""
    def __init__(self, path, tokenizer, examples, device, milestones):
        super().__init__(path,[tokenizer.convert_tokens_to_ids(x) for x in ['<extra_id_0>','<extra_id_1>']])
        self.collator=BatchCollator(tokenizer)
        self.examples=examples
        self.device=device
        self.milestones=set(milestones)
        self.loss_path=path.parent/'teacher-loss.jsonl'
        self.stage='substructure'
        self.epoch=1

    def measure(self, model):
        """Evaluate the same eight TRAIN targets with full gold prefixes."""
        was_training=model.training
        try:
            with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
                model.eval()
                losses=[]
                with torch.no_grad():
                    for example in self.examples:
                        batch=self.collator([example])
                        values={k:v.to(self.device) for k,v in batch.items() if k!='ids'}
                        value=model(**values,use_cache=False).loss
                        if not torch.isfinite(value):raise FloatingPointError('Nonfinite teacher loss')
                        losses.append(value.item())
        finally:
            model.train(was_training)
        record={'optimizer_update':self.step,'ids':[x['id'] for x in self.examples],
                'mean_microbatch_teacher_loss':sum(losses)/len(losses),'sample_losses':losses}
        with self.loss_path.open('a') as handle:handle.write(json.dumps(record)+'\n')
        print(json.dumps({'phase':'FIXED_TRAIN_LOSS',**record}),flush=True)

    def after_step(self, model):
        super().after_step(model)
        if self.step in self.milestones:self.measure(model)


def run(bundle, config, recipe, output):
    """Initialize pretrained mT5, train all TRAIN once, snapshot, then evaluate8DEV."""
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    output.mkdir(parents=True,exist_ok=False)
    if shutil.disk_usage(output).free<12*1024**3:raise OSError('Need12GiB for source cache and snapshots')
    def save(name,value):(output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2))
    trace=EvaluationTrace(output/'trace.jsonl')
    run_signature={'implementation':'text2event-clean-pilot-v1',**recipe,'config':config,
                   'gradient_checkpointing':True,'files':bundle_hashes(bundle)}
    save('request.json',{'kind':'fresh-one-epoch-substructure-diagnostic','run':run_signature,
         'checkpoint_resumed':False,'dev_samples':8,'full_stage_started':False,'test_used':False})
    random.seed(recipe['seed']);torch.manual_seed(recipe['seed'])
    trace.emit('MODEL_LOAD_START')
    model,t=load_components(config,bundle/'tokenizer');model.to('cuda')
    save('model-loading.json',model.text2event_load_report)
    model.gradient_checkpointing_enable();model.config.use_cache=False
    folder=bundle/'GENEVA'
    ds=BundleDataset(folder,'train',t,config['max_source_length'],config['max_target_length'],
                     'substructure',config['substructure_source_prefix'])
    loader=DataLoader(ds,batch_size=recipe['micro_batch'],shuffle=True,collate_fn=BatchCollator(t))
    optimizer=torch.optim.AdamW(model.parameters(),lr=recipe['lr'],weight_decay=0.)
    scheduler,schedule=build_scheduler(optimizer,len(loader),recipe['accumulation'],recipe['substructure_epochs'],
                                      recipe['scheduler'],warmup_ratio=recipe['warmup_ratio'])
    save('schedule.json',schedule)
    indices=sorted(random.Random(421).sample(range(len(ds)),8))
    observer=PilotObserver(output/'optimizer-updates.jsonl',t,[ds[i] for i in indices],
                           'cuda',{1,8,schedule['warmup_updates'],64,118})
    observer.measure(model)
    trace.emit('TRAIN_START',epoch=1,samples=len(ds),microbatches=len(loader),initial_lr=scheduler.get_last_lr())
    started=time.perf_counter()
    metrics=train_epoch(model,loader,optimizer,scheduler,'cuda',recipe['accumulation'],
                        progress_every=64,step_observer=observer)
    metrics['train_seconds']=time.perf_counter()-started
    save('train-epoch-1.json',metrics)
    trace.emit('TRAIN_END',epoch=1,**metrics)
    signature={'run':run_signature,'stage':'substructure'}
    trace.emit('SNAPSHOT_START',epoch=1)
    save_checkpoint(output/'pre-dev-state.pt',model,optimizer,scheduler,1,-1.,signature)
    save_best(output/'pre-dev-weights.pt',model,1,-1.,signature)
    save('snapshot.json',{'epoch':1,'phase':'before_DEV','dev_evaluated':False,
         'scheduler_update':scheduler.last_epoch,'state':'pre-dev-state.pt','weights':'pre-dev-weights.pt',
         'best_field_is_selection_score':False})
    trace.emit('SNAPSHOT_SAVED',epoch=1,scheduler_update=scheduler.last_epoch)
    del optimizer,scheduler,loader,ds
    torch.cuda.empty_cache()
    dev=list(map(json.loads,(folder/'dev.gold.jsonl').read_text().splitlines()))
    indices=sorted(random.Random(421).sample(range(len(dev)),8));samples=[dev[i] for i in indices]
    save('evaluation-request.json',{'ids':[x['id'] for x in samples],'config':config,'split':'dev','samples':8})
    trace.emit('DEV_START',samples=8)
    began=time.perf_counter()
    scores,issues=evaluate_substructure(model,t,samples,folder/'event.schema',config,'cuda',observer=trace)
    save('diagnostics.json',issues)
    save('summary.json',{'scores':scores,'seconds':time.perf_counter()-began,'samples':8,
         'invalid':sum(x['status']!='ok' for x in issues),
         'true_empty':sum(x['status']=='ok' and x.get('generated','').replace(' ','')=='<extra_id_0><extra_id_1>' for x in issues)})
    save('completion.json',{'status':'complete','substructure_epochs_added':1,'optimizer_updates':metrics['optimizer_updates'],
         'full_DEV_completed':False,'full_stage_started':False,'test_used':False})
    trace.emit('PILOT_COMPLETE')
