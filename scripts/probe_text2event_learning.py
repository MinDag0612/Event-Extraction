"""Fixed TRAIN memorization probe. Gate full training on both substructure skills."""
import argparse
import json
import random
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from src.baselines.text2event.training import load_components,BatchCollator,train_epoch
from src.baselines.text2event.convert import convert,convert_substructure
from src.baselines.text2event.curriculum import evaluate_substructure
from src.baselines.text2event.evaluation import evaluate_model
from src.baselines.text2event.learning_probe import teacher_probe,component_scores,gate_passed
from src.baselines.text2event.diagnostics import generation_summary
from src.baselines.text2event.pilot_utils import save_weights_atomic


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--dataset',choices=['BKEE','GENEVA','PHEE','VHE'],required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--examples',type=int,default=8)
    p.add_argument('--substructure-epochs',type=int,default=80)
    p.add_argument('--full-epochs',type=int,default=40)
    p.add_argument('--eval-every',type=int,default=8)
    p.add_argument('--threshold',type=float,default=.8)
    p.add_argument('--device',default='cuda')
    a=p.parse_args()
    if min(a.examples,a.substructure_epochs,a.full_epochs,a.eval_every)<1 or not 0<a.threshold<=1:p.error('Invalid settings')
    a.output.mkdir(parents=True,exist_ok=False)
    config=json.loads(Path('configs/text2event/preparation-upstream.json').read_text())
    config.update(constraint_decoder='upstream-compatible',empty_root_penalty=0.)
    random.seed(421);torch.manual_seed(421)
    folder=a.bundle/a.dataset
    rows=list(map(json.loads,(folder/'train.gold.jsonl').read_text().splitlines()))
    positives=sorted([r for r in rows if any(e['arguments'] for e in r['events'])],key=lambda r:len(convert_substructure(r)['event']))
    negatives=sorted([r for r in rows if not r['events']],key=lambda r:len(r['tokens']))
    nneg=min(1,len(negatives),a.examples-1)
    selected=positives[:a.examples-nneg]+negatives[:nneg]
    if len(selected)!=a.examples:raise ValueError('Need enough TRAIN positives with arguments')
    (a.output/'selected-train.json').write_text(json.dumps(selected,ensure_ascii=False,indent=2)+'\n')
    (a.output/'run.json').write_text(json.dumps({'kind':'TRAIN-only-learning-probe-not-baseline','dataset':a.dataset,
        'examples':a.examples,'substructure_epochs':a.substructure_epochs,'full_epochs':a.full_epochs,
        'eval_every':a.eval_every,'threshold':a.threshold,'lr':1e-4,'seed':421,'batch':1,
        'selection':'gate passed, minimum component F1, combined F1, then minimum teacher-forced token loss','config':config},indent=2)+'\n')
    model,t=load_components(config,a.bundle/'tokenizer');model.to(a.device)
    model.gradient_checkpointing_enable();model.config.use_cache=False
    (a.output/'model-loading.json').write_text(json.dumps(model.text2event_load_report)+'\n')
    types,roles,_=map(json.loads,(folder/'event.schema').read_text().splitlines())
    passed=False;best_key=None;sub_completed=full_completed=0
    for stage,converter,epochs in [('substructure',convert_substructure,a.substructure_epochs),('full',convert,a.full_epochs)]:
        if stage=='full':
            if not passed:break
            model.load_state_dict(torch.load(a.output/'best-substructure.pt',map_location='cpu',weights_only=False)['model'])
            (a.output/'best-substructure.pt').unlink()
        prefix=config['substructure_source_prefix' if stage=='substructure' else 'full_source_prefix']
        examples=[]
        for row in selected:
            record=converter(row);source=t(prefix+record['text'],truncation=False);target=t(record['event'],truncation=False)['input_ids']
            if len(source['input_ids'])>config['max_source_length'] or len(target)>config['max_target_length']:raise ValueError('Length overflow')
            examples.append(dict(id=row['id'],input_ids=source['input_ids'],attention_mask=source['attention_mask'],labels=target))
        (a.output/f'{stage}-tokenization.json').write_text(json.dumps([{'id':e['id'],'input_ids':e['input_ids'],
            'target_ids':e['labels'],'decoded_input':t.decode(e['input_ids'],skip_special_tokens=False),
            'decoded_target':t.decode(e['labels'],skip_special_tokens=False)} for e in examples],ensure_ascii=False,indent=2)+'\n')
        loader=DataLoader(examples,batch_size=1,shuffle=True,collate_fn=BatchCollator(t))
        opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=0.)
        scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.)
        for epoch in range(epochs+1):
            began=time.monotonic()
            metrics=train_epoch(model,loader,opt,scheduler,a.device,progress_every=a.examples) if epoch else {}
            if stage=='substructure':sub_completed=epoch
            else:full_completed=epoch
            metrics['train_seconds']=time.monotonic()-began
            if epoch%a.eval_every and epoch!=epochs:continue
            # Preserve weights before potentially expensive generation (weights-only).
            disk=save_weights_atomic(a.output/'latest-weights.pt',model,stage=stage,epoch=epoch)
            print(json.dumps({'checkpoint':stage,'epoch':epoch,'disk':disk}),flush=True)
            teacher=teacher_probe(model,t,examples,a.device)
            (a.output/f'{stage}-{epoch}-teacher.json').write_text(json.dumps(teacher,ensure_ascii=False,indent=2)+'\n')
            print(json.dumps({'stage':stage,'epoch':epoch,'teacher_loss':teacher['mean_token_loss'],
                              'teacher_token_accuracy':teacher['token_accuracy'],'generation':'starting'}),flush=True)
            eval_start=time.monotonic()
            if stage=='substructure':
                score,issues=evaluate_substructure(model,t,selected,folder/'event.schema',config,a.device)
                components=component_scores(selected,issues,types,roles,t.backend_tokenizer.normalizer.normalize_str)
                metrics['components']=components;passed=gate_passed(components,a.threshold) and score['f1']>=a.threshold
                key=(int(passed),min(components[k]['f1'] for k in ['type_trigger','role_argument']),score['f1'],-teacher['mean_token_loss'])
                if best_key is None or key>best_key:
                    best_key=key
                    save_weights_atomic(a.output/'best-substructure.pt',model,stage=stage,epoch=epoch)
                    (a.output/'best-substructure.json').write_text(json.dumps({'epoch':epoch,'key':key,'gate_passed':passed})+'\n')
            else:
                score,pred,issues=evaluate_model(model,t,selected,folder/'event.schema',config,a.device)
                metrics['generation']=generation_summary(pred,issues)
            metrics.update(stage=stage,epoch=epoch,scores=score,diagnostics=issues,
                teacher_loss=teacher['mean_token_loss'],teacher_token_accuracy=teacher['token_accuracy'],
                eval_seconds=time.monotonic()-eval_start)
            (a.output/f'{stage}-{epoch}.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2)+'\n')
            print(json.dumps({k:v for k,v in metrics.items() if k!='diagnostics'}),flush=True)
            if stage=='substructure' and passed:break
    completion={'kind':'TRAIN-only-probe-not-baseline','substructure_gate_passed':passed,
                'substructure_epochs_completed':sub_completed,'full_epochs_completed':full_completed,
                'full_started':passed,'threshold':a.threshold}
    (a.output/'completion.json').write_text(json.dumps(completion,indent=2)+'\n')
    print(json.dumps(completion),flush=True)


if __name__=='__main__':main()
