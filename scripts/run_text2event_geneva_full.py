"""Frozen VHE trainer, epoch-sized GENEVA runs and DEV-verified final TEST."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--lock',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    spec=spec_from_file_location('common',Path(__file__).with_name('run_text2event_short_vhe.py'))
    common=module_from_spec(spec);spec.loader.exec_module(common)
    lock=json.loads(a.lock.read_text());config=common.validate_source(a.source,lock)
    folder=a.source/'bundle/GENEVA'
    gold={s:list(map(json.loads,(folder/(s+'.gold.jsonl')).read_text().splitlines())) for s in ['train','dev','test']}
    if [len(gold[s]) for s in ['train','dev','test']] != [1874,766,930]:raise ValueError('GENEVA split counts differ')
    command=[sys.executable,'-u','-m','scripts.train_text2event','--bundle',str(a.source/'bundle'),
             '--dataset','GENEVA','--config',str(a.source/'configs/text2event/baseline-mt5.json'),
             '--output',str(a.output),'--epochs','30','--substructure-epochs','5','--micro-batch','1',
             '--accumulation','16','--lr','0.0001','--lr-scheduler','linear','--warmup-steps','0',
             '--warmup-ratio','0.05','--seed','421','--patience','3','--min-delta','0.001',
             '--min-full-epochs','5','--gradient-checkpointing','--max-epochs-this-run','1',
             '--prune-completed-state','--progress-every','64']
    if (a.output/'run.json').exists():command.append('--resume')
    print('GENEVA_FROZEN_PREFLIGHT_OK',flush=True)
    subprocess.run(command,cwd=a.source,check=True)
    if not (a.output/'completion.json').exists():
        invocation=json.loads((a.output/'invocation.json').read_text())
        write(a.output/'status.json',{'status':'epoch_complete','dev_complete':True,'dev_samples':766,**invocation})
        return
    # Training has stopped; reload only selected weights, verify DEV before opening TEST.
    sys.path.insert(0,str(a.source))
    import torch
    from src.baselines.text2event.training import load_components
    from src.baselines.text2event.evaluation import evaluate_model
    state=torch.load(a.output/'full/best.pt',map_location='cpu',weights_only=False)
    run=json.loads((a.output/'run.json').read_text())
    if state['signature']!={'run':run,'stage':'full'}:raise ValueError('Best signature differs')
    model,tokenizer=load_components(config,a.source/'bundle/tokenizer');model.load_state_dict(state['model']);model.to('cuda');del state
    completion=json.loads((a.output/'completion.json').read_text());epoch=completion['best_full_epoch']
    scores,pred,issues=evaluate_model(model,tokenizer,gold['dev'],folder/'event.schema',config,'cuda')
    expected=json.loads((a.output/f'full/epoch-{epoch}-metrics.json').read_text())['dev']
    if scores!=expected:raise ValueError('Selected DEV reproduction differs; TEST remains unopened')
    write(a.output/'dev-results.json',{'checkpoint_epoch':epoch,'samples':766,'scores':scores,'diagnostics':issues})
    write(a.output/'dev-predictions.json',pred)
    scores,pred,issues=evaluate_model(model,tokenizer,gold['test'],folder/'event.schema',config,'cuda')
    write(a.output/'test-results.json',{'checkpoint_epoch':epoch,'samples':930,'scores':scores,'diagnostics':issues})
    write(a.output/'test-predictions.json',pred)
    completion.update(status='complete',baseline_complete=True,dev_verified=True,test_evaluated=True,test_used_for_selection=False,test_samples=930,test_scores=scores)
    write(a.output/'completion.json',completion);write(a.output/'status.json',completion)


if __name__=='__main__':main()
