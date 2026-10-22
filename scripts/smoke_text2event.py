"""Real mT5 smoke, distinct from baseline training. Requires explicit invocation."""
import argparse
import json
import random
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from src.baselines.text2event.training import load_components,BundleDataset,BatchCollator,train_epoch
from src.baselines.text2event.checkpoint import save_checkpoint,restore_checkpoint
from src.baselines.text2event.evaluation import evaluate_model


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--dataset',choices=['BKEE','GENEVA','PHEE','VHE'],required=True)
    p.add_argument('--config',type=Path,default=Path('configs/text2event/preparation.json'))
    p.add_argument('--device',default='cuda')
    p.add_argument('--stress-max-length',action='store_true')
    a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    if a.device=='cuda' and not torch.cuda.is_available(): raise RuntimeError('Enable a Kaggle GPU')
    a.output.mkdir(parents=True)
    config=json.loads(a.config.read_text());random.seed(42);torch.manual_seed(42)
    report={'dataset':a.dataset,'kind':'smoke-not-baseline','status':'running','config':config,
            'torch':torch.__version__,'cuda':torch.version.cuda,'device':a.device}
    began=time.monotonic()
    try:
        model,tokenizer=load_components(config,a.bundle/'tokenizer');model.to(a.device)
        model.gradient_checkpointing_enable();model.config.use_cache=False
        if a.device=='cuda':
            torch.cuda.reset_peak_memory_stats()
            report['gpu']=torch.cuda.get_device_name()
        collator=BatchCollator(tokenizer)
        opt=torch.optim.AdamW(model.parameters(),lr=1e-4)
        scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda step:max(0.,1-step/4))
        signature={'dataset':a.dataset,'config':config,'kind':'smoke'}
        stages={}
        for stage in ['trigger','full']:
            ds=BundleDataset(a.bundle/a.dataset,'train',tokenizer,config['max_source_length'],config['max_target_length'],stage)
            # Two real examples, including the longest target in TRAIN.
            indices=list(dict.fromkeys([0,max(range(len(ds)),key=lambda i:len(ds[i]['labels']))]))
            rows=[ds[i] for i in indices]
            loader=DataLoader(rows,batch_size=1,collate_fn=collator)
            stages[stage]=train_epoch(model,loader,opt,scheduler,a.device,accumulation=2)
        report['training']=stages
        if a.stress_max_length:
            # Synthetic length stress only, not an example used in evaluation.
            item=rows[-1]
            def extend(values,length):return (values* ((length+len(values)-1)//len(values)))[:length-1]+[tokenizer.eos_token_id]
            stress={'id':'synthetic-length-stress','input_ids':extend(item['input_ids'],config['max_source_length']),
                    'attention_mask':[1]*config['max_source_length'],
                    'labels':extend(item['labels'],config['max_target_length'])}
            report['length_stress']=train_epoch(model,DataLoader([stress],collate_fn=collator),opt,scheduler,a.device)
        save_checkpoint(a.output/'last.pt',model,opt,scheduler,1,-1,signature)
        before_step=scheduler.last_epoch
        expected_torch=torch.rand(4);expected_python=random.random()
        # Perturb state so restore actually has to change it.
        scheduler.step();random.seed(123);torch.manual_seed(123)
        epoch,best=restore_checkpoint(a.output/'last.pt',model,opt,scheduler,signature)
        if scheduler.last_epoch!=before_step or epoch!=1 or not torch.equal(expected_torch,torch.rand(4)) or random.random()!=expected_python:
            raise AssertionError('Resume state mismatch')
        report['resume']='PASS: scheduler/epoch/RNG restored'
        report['post_resume_update']=train_epoch(model,DataLoader([rows[0]],collate_fn=collator),opt,scheduler,a.device)
        gold=[json.loads(l) for l in (a.bundle/a.dataset/'dev.gold.jsonl').read_text().splitlines()][:2]
        scores,predictions,diagnostics=evaluate_model(model,tokenizer,gold,a.bundle/a.dataset/'event.schema',config,a.device)
        report['scores_not_baseline']=scores
        report['generation_statuses']=[r['status'] for r in diagnostics]
        (a.output/'predictions.json').write_text(json.dumps(predictions,ensure_ascii=False,indent=2)+'\n')
        (a.output/'diagnostics.json').write_text(json.dumps(diagnostics,ensure_ascii=False,indent=2)+'\n')
        # Low quality after a few updates is expected; malformed generation is a separate gate.
        report['status']='PASS' if all(r['status']=='ok' for r in diagnostics) else 'REVIEW_GENERATION'
    except torch.OutOfMemoryError as exc:
        report.update(status='OOM',error=str(exc));raise
    except Exception as exc:
        report.update(status='FAILED',error=f'{type(exc).__name__}: {exc}');raise
    finally:
        report['seconds']=time.monotonic()-began
        if a.device=='cuda' and torch.cuda.is_available():report['peak_vram_bytes']=torch.cuda.max_memory_allocated()
        (a.output/'smoke-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
