"""Text2Event-mT5 trainer: explicit trigger-first curriculum variant; epoch-boundary resume."""
import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from src.baselines.text2event.training import load_components, BundleDataset, BatchCollator, train_epoch
from src.baselines.text2event.evaluation import evaluate_model
from src.baselines.text2event.checkpoint import save_checkpoint, restore_checkpoint, save_best


from src.baselines.text2event.early_stopping import EarlyStopping


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--dataset',choices=['BKEE','GENEVA','PHEE','VHE'],required=True)
    p.add_argument('--config',type=Path,default=Path('configs/text2event/preparation.json'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=30)
    p.add_argument('--trigger-epochs',type=int,default=1,help='Variant: trigger-only warmup, then full trees; 0 disables')
    p.add_argument('--patience',type=int,default=5)
    p.add_argument('--min-delta',type=float,default=0.0,help='DEV AC units 0..1; 0.001 = 0.1 percentage point')
    p.add_argument('--micro-batch',type=int,default=1)
    p.add_argument('--accumulation',type=int,default=8)
    p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--gradient-checkpointing',action='store_true')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--final-evaluation',action='store_true',help='Evaluate TRAIN/TEST after early stopping or max epochs and DEV selection')
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    a=p.parse_args()
    if min(a.epochs,a.micro_batch,a.accumulation)<1 or a.lr<=0 or not 0<=a.trigger_epochs<a.epochs:
        p.error('Invalid training arguments')
    stopper=EarlyStopping(a.patience,a.min_delta)
    if a.output.exists() and not a.resume: raise FileExistsError(a.output)
    if a.resume and not (a.output/'last.pt').exists(): raise FileNotFoundError('Missing last.pt')
    random.seed(a.seed);torch.manual_seed(a.seed)
    config=json.loads(a.config.read_text());folder=a.bundle/a.dataset
    # Pin content, tokenizer and optimization schedule, not local absolute paths.
    signature={'config':config,'dataset':a.dataset,'epochs':a.epochs,'trigger_epochs':a.trigger_epochs,
               'batch':a.micro_batch,'accumulation':a.accumulation,'lr':a.lr,'seed':a.seed,
               'gradient_checkpointing':a.gradient_checkpointing,
               'early_stopping':{'patience':a.patience,'min_delta':a.min_delta,'start':'full-stage'},
               'files':{str(f.relative_to(a.bundle)):hashlib.sha256(f.read_bytes()).hexdigest()
                        for f in list(folder.glob('*.json'))+list(folder.glob('*.jsonl'))+[folder/'event.schema']+list((a.bundle/'tokenizer').glob('*')) if f.is_file()}}
    model,tokenizer=load_components(config,a.bundle/'tokenizer')
    datasets={stage:BundleDataset(folder,'train',tokenizer,config['max_source_length'],config['max_target_length'],stage)
              for stage in ['trigger','full']}
    loaders={stage:DataLoader(ds,batch_size=a.micro_batch,shuffle=True,collate_fn=BatchCollator(tokenizer)) for stage,ds in datasets.items()}
    dev=[json.loads(l) for l in (folder/'dev.gold.jsonl').read_text().splitlines()]
    model.to(a.device)
    if a.gradient_checkpointing:
        model.gradient_checkpointing_enable();model.config.use_cache=False
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr)
    steps=math.ceil(len(loaders['full'])/a.accumulation)*a.epochs
    scheduler=get_linear_schedule_with_warmup(optimizer,num_warmup_steps=int(.1*steps),num_training_steps=steps)
    start=0;best=-1.
    if a.resume: start,best=restore_checkpoint(a.output/'last.pt',model,optimizer,scheduler,signature,early_stopping=stopper)
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'run.json').write_text(json.dumps(signature,indent=2)+'\n')
    completed_epoch=start
    for epoch in range(start,a.epochs):
        if stopper.stopped: break
        stage='trigger' if epoch<a.trigger_epochs else 'full';began=time.monotonic()
        if a.device.startswith('cuda'): torch.cuda.reset_peak_memory_stats()
        metrics=train_epoch(model,loaders[stage],optimizer,scheduler,a.device,a.accumulation)
        scores,predictions,issues=evaluate_model(model,tokenizer,dev,folder/'event.schema',config,a.device)
        metrics.update(epoch=epoch+1,stage=stage,dev=scores,seconds=time.monotonic()-began,
                       invalid_predictions=sum(x['status']!='ok' for x in issues),
                       peak_vram_bytes=torch.cuda.max_memory_allocated() if a.device.startswith('cuda') else None)
        stopper.update(scores['AC']['f1'],active=(stage=='full'))
        metrics['early_stopping']=stopper.state_dict()
        if scores['AC']['f1']>best:
            best=scores['AC']['f1'];save_best(a.output/'best.pt',model,epoch+1,best,signature)
        save_checkpoint(a.output/'last.pt',model,optimizer,scheduler,epoch+1,best,signature,early_stopping=stopper)
        (a.output/f'epoch-{epoch+1}.json').write_text(json.dumps(metrics,indent=2)+'\n')
        (a.output/'dev-predictions.json').write_text(json.dumps(predictions,ensure_ascii=False)+'\n')
        (a.output/'dev-diagnostics.json').write_text(json.dumps(issues,ensure_ascii=False)+'\n')
        completed_epoch=epoch+1
        print(metrics,flush=True)
        if stopper.stopped: break
    (a.output/'completion.json').write_text(json.dumps({'epochs_completed':completed_epoch,
        'max_epochs':a.epochs,'stop_reason':'early_stopping' if stopper.stopped else 'max_epochs',
        'early_stopping':stopper.state_dict(),'best_dev_ac':best},indent=2)+'\n')
    if a.final_evaluation:
        state=torch.load(a.output/'best.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(state['model'])
        for split in ['train','test']:
            gold=[json.loads(l) for l in (folder/f'{split}.gold.jsonl').read_text().splitlines()]
            scores,pred,issues=evaluate_model(model,tokenizer,gold,folder/'event.schema',config,a.device)
            (a.output/f'{split}-results.json').write_text(json.dumps({'checkpoint_epoch':state['epoch'],'scores':scores,'diagnostics':issues},ensure_ascii=False,indent=2)+'\n')
            (a.output/f'{split}-predictions.json').write_text(json.dumps(pred,ensure_ascii=False)+'\n')
        model.save_pretrained(a.output/'best-model');tokenizer.save_pretrained(a.output/'best-model')

if __name__=='__main__': main()
