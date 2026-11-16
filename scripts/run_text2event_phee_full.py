"""Continue frozen PHEE training with full DEV selection and final TEST evaluation."""
import argparse
import json
import random
import sys
import time
from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec


def helper(path, name):
    spec = spec_from_file_location(name, path)
    module = module_from_spec(spec); spec.loader.exec_module(module)
    return module


common = helper(Path(__file__).with_name('run_text2event_short_vhe.py'), 'short_common')
write, digest = common.write, common.digest


def validate_state(state, previous, metadata, schedule):
    required = {'model', 'optimizer', 'scheduler', 'signature', 'epoch', 'best', 'python_rng', 'torch_rng', 'cuda_rng'}
    if not required <= state.keys() or not state['optimizer'].get('state'):
        raise ValueError('Expected full optimizer/scheduler/RNG state')
    if state['signature'] != {'run': previous, 'stage': 'full'}:
        raise ValueError('Unexpected source signature or stage')
    epoch = metadata['source_epoch']
    if metadata['mode'] == 'final-evaluation':
        stopping = helper(Path(__file__).with_name('text2event_phee_stopping.py'), 'full_stopping')
        if not stopping.permits_final_evaluation(epoch, metadata.get('early_stopping')):
            raise ValueError('TEST is allowed only after full epoch30 or validated early stopping')
    if state['epoch'] != epoch or state['scheduler']['last_epoch'] != epoch * 179:
        raise ValueError('Source epoch or scheduler cursor differs')
    if schedule != {'kind': 'linear', 'total_optimizer_updates': 5370, 'warmup_updates': 269}:
        raise ValueError('Full 30-epoch schedule differs')
    if not state['cuda_rng']:
        raise ValueError('Missing CUDA RNG')
    if metadata['mode'] == 'bootstrap':
        if epoch != 3 or previous['pilot']['full_epochs'] != 1:
            raise ValueError('Expected original epoch3 short-pilot signature')
    elif previous.get('protocol', {}).get('selection_scope') != 'full DEV949 from epoch3':
        raise ValueError('Expected full-DEV continuation protocol')


def worker(a):
    output = a.output; output.mkdir(parents=True, exist_ok=True)
    def event(phase, **fields):
        value = {'phase': phase, 'time': time.time(), **fields}
        write(output/'activity.json', value); print(json.dumps(value), flush=True)
    event('PREFLIGHT')
    source = a.source.resolve(); metadata = json.loads(a.metadata.read_text()); lock = json.loads(a.lock.read_text())
    config = common.validate_source(source, lock)
    previous = json.loads(a.previous_run.read_text())
    if previous['dataset'] != 'PHEE' or previous['config'] != config:
        raise ValueError('Dataset/model differs')
    for key, value in lock['trainer'].items():
        if previous[key] != value: raise ValueError('Training recipe differs: '+key)
    for name, expected in previous['files'].items():
        if digest(source/'bundle'/name) != expected: raise ValueError('Bundle differs: '+name)
    signature = {key: value for key, value in previous.items() if key not in {'pilot', 'protocol'}}
    signature['protocol'] = {'selection_scope': 'full DEV949 from epoch3', 'target_full_epochs': 30,
                             'inherited_substructure_selection': 'DEV32, epoch5 from pilot', 'test_selection': False}
    stage_signature = {'run': signature, 'stage': 'full'}
    write(output/'run.json', signature); write(output/'source-metadata.json', {k:v for k,v in metadata.items() if k!='downloads'})
    sys.path.insert(0, str(source))
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer
    from scripts.train_text2event import build_scheduler
    from src.baselines.text2event.training import load_components, BundleDataset, BatchCollator, train_epoch
    from src.baselines.text2event.checkpoint import save_checkpoint, save_best
    from src.baselines.text2event.evaluation import evaluate_model, score_by_id
    import inspect
    for fn in [build_scheduler, train_epoch, evaluate_model]:
        if not Path(inspect.getfile(inspect.unwrap(fn))).resolve().is_relative_to(source):
            raise ValueError('Import escaped frozen source')
    folder = source/'bundle/PHEE'; tokenizer = AutoTokenizer.from_pretrained(source/'bundle/tokenizer', local_files_only=True)
    dev = list(map(json.loads, (folder/'dev.gold.jsonl').read_text().splitlines()))
    if len(dev)!=949: raise ValueError('Expected full DEV949')
    ds = BundleDataset(folder, 'train', tokenizer, config['max_source_length'], config['max_target_length'], 'full', config['full_source_prefix'])
    checked = BundleDataset(folder, 'dev', tokenizer, config['max_source_length'], config['max_target_length'], 'full', config['full_source_prefix'])
    if len(ds)!=2853 or len(checked)!=949: raise ValueError('Unexpected split counts')
    write(output/'preflight.json', {'train': len(ds), 'dev': len(checked), 'source_manifest_sha256': lock['source_manifest_sha256']})
    del checked
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    event('MODEL_LOAD'); model, tokenizer = load_components(config, source/'bundle/tokenizer'); model.to('cuda')
    model.gradient_checkpointing_enable(); model.config.use_cache=False
    loader = DataLoader(ds, batch_size=1, shuffle=True, collate_fn=BatchCollator(tokenizer))
    optimizer = torch.optim.AdamW(model.parameters(), lr=signature['lr'], weight_decay=0.)
    scheduler, schedule = build_scheduler(optimizer, len(loader), 16, 30, 'linear', 0, 0.05)
    write(output/'full/schedule.json', schedule)
    event('CHECKPOINT_RESTORE'); state = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    validate_state(state, previous, metadata, schedule)
    if len(state['cuda_rng'])!=torch.cuda.device_count(): raise ValueError('CUDA device count differs')
    model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler'])
    random.setstate(state['python_rng']); torch.set_rng_state(state['torch_rng']); torch.cuda.set_rng_state_all(state['cuda_rng'])
    epoch=state['epoch']; del state
    commit = json.loads(a.commit.read_text())
    if commit['epoch']!=epoch or commit.get('scheduler_update')!=179*epoch: raise ValueError('Commit cursor differs')
    if metadata['mode']=='bootstrap':
        if commit.get('selection_scope')!='fixed DEV subset' or commit['stage']!='full': raise ValueError('Expected pilot commit')
        best=-1.; best_epoch=None; best_dev_scores=None
    else:
        if commit.get('selection_scope')!='full DEV949' or not commit.get('dev_complete'): raise ValueError('Incomplete full DEV commit')
        best=commit['best_full_dev']; best_epoch=commit['best_epoch']; best_dev_scores=commit['best_dev_scores']
    restore={'epoch':epoch,'scheduler_update':scheduler.last_epoch,'optimizer_entries':len(optimizer.state),
             'selection_scope':signature['protocol']['selection_scope'],'best_full_dev':best,
             'subset_score_not_reused_as_full_dev':True}
    write(output/'restore.json', restore); event('RESTORED', **restore)
    checkpoint=output/'full/pre-dev-state.pt'
    if metadata['mode']=='bootstrap':
        event('CHECKPOINT_SAVE'); save_checkpoint(checkpoint, model, optimizer, scheduler, epoch, best, stage_signature)
        a.checkpoint.unlink()
    else:
        a.checkpoint.replace(checkpoint)
        best_state=torch.load(a.best_checkpoint, map_location='cpu', weights_only=False)
        if best_state['signature']!=stage_signature or best_state['epoch']!=best_epoch or best_state['best']!=best:
            raise ValueError('Full DEV best weights metadata differs')
        del best_state; a.best_checkpoint.replace(output/'full/best.pt')
        write(output/f'full/epoch-{epoch}-evaluation-commit.json', commit)
    def evaluate(rows, split, current_epoch):
        event('EVALUATION_START', split=split, epoch=current_epoch, total=len(rows))
        predictions=[]; issues=[]; began=time.monotonic(); python_rng=random.getstate()
        with torch.random.fork_rng(), (output/f'full/epoch-{current_epoch}-{split}.jsonl').open('w') as stream:
            for index, sample in enumerate(rows,1):
                event('SAMPLE_START', split=split, epoch=current_epoch, index=index, total=len(rows), id=sample['id'])
                _,pred,one=evaluate_model(model,tokenizer,[sample],folder/'event.schema',config,'cuda')
                predictions.extend(pred);issues.extend(one)
                stream.write(json.dumps({'prediction':pred[0],'diagnostic':one[0]},ensure_ascii=False)+'\n');stream.flush()
                event('SAMPLE_DONE', split=split, epoch=current_epoch,index=index,status=one[0]['status'])
        random.setstate(python_rng)
        scores=score_by_id(predictions,rows)
        result={'epoch':current_epoch,'split':split,'samples':len(rows),'scores':scores,
                'diagnostics':common.diagnostic_summary(issues),'seconds':time.monotonic()-began}
        write(output/f'full/epoch-{current_epoch}-{split}-results.json',result)
        print(json.dumps(result),flush=True)
        return result
    def dev_commit(current_epoch):
        nonlocal best,best_epoch,best_dev_scores
        result=evaluate(dev,'dev',current_epoch)
        if result['scores']['AC']['f1']>best:
            best=result['scores']['AC']['f1'];best_epoch=current_epoch;best_dev_scores=result['scores']
            event('CHECKPOINT_SAVE');save_best(output/'full/best.pt',model,current_epoch,best,stage_signature)
        record={'stage':'full','epoch':current_epoch,'scheduler_update':scheduler.last_epoch,'selection_scope':'full DEV949',
                'dev_complete':True,'best_full_dev':best,'best_epoch':best_epoch,'best_dev_scores':best_dev_scores,'results':result}
        write(output/f'full/epoch-{current_epoch}-evaluation-commit.json',record)
        return record
    if metadata['mode']=='bootstrap': commit=dev_commit(epoch)
    if metadata['mode']=='final-evaluation':
        event('BEST_MODEL_LOAD');saved=torch.load(output/'full/best.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(saved['model']);del saved,optimizer,scheduler
        # Verify selected DEV before the single final TEST pass.
        reproduced=evaluate(dev,'selected-dev',best_epoch)
        expected=metadata['selected_dev_scores']
        if reproduced['scores']!=expected: raise ValueError('Selected DEV reproduction differs; TEST not started')
        test=list(map(json.loads,(folder/'test.gold.jsonl').read_text().splitlines()))
        if len(test)!=961:raise ValueError('Expected TEST961')
        final=evaluate(test,'test',best_epoch)
        write(output/'completion.json',{'status':'complete','baseline_complete':True,'full_epochs_completed':epoch,'early_stopping':metadata.get('early_stopping'),
               'best_epoch':best_epoch,'selection_scope':'full DEV949 from epoch3','inherited_substructure_selection':'DEV32',
               'dev_verified':True,'test_evaluated':True,'test_used_for_selection':False,'test_scores':final['scores']})
        write(output/'status.json',{'status':'complete','baseline_complete':True,'epoch':epoch,'best_epoch':best_epoch,'early_stopping':metadata.get('early_stopping'),'test_evaluated':True})
        return
    class Observer:
        def before_step(self,model,optimizer,gradient_norm):pass
        def after_step(self,model):event('TRAIN_PROGRESS',epoch=epoch,optimizer_update=scheduler.last_epoch)
    target=min(epoch+1,30)
    if epoch<30:
        epoch=target;event('TRAIN_PROGRESS',epoch=epoch,optimizer_update=scheduler.last_epoch)
        began=time.monotonic();metrics=train_epoch(model,loader,optimizer,scheduler,'cuda',16,progress_every=64,step_observer=Observer())
        metrics['seconds']=time.monotonic()-began;write(output/f'full/epoch-{epoch}-train.json',metrics)
        event('CHECKPOINT_SAVE');save_checkpoint(checkpoint,model,optimizer,scheduler,epoch,best,stage_signature)
        write(output/'full/snapshot.json',{'stage':'full','epoch':epoch,'scheduler_update':scheduler.last_epoch,
              'phase':'before_DEV','previous_best_full_dev':best,'resume_protocol':'pre-dev-state plus matching full-DEV evaluation commit'})
        commit=dev_commit(epoch)
    write(output/'status.json',{'status':'epoch_complete','epoch':epoch,'full_epochs_target':30,'dev_complete':True,
          'best_full_dev':best,'best_epoch':best_epoch,'baseline_complete':False,'test_evaluated':False,
          'next_action':'final-evaluation' if epoch==30 else 'continue','automatic_continuation_authorized':True})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['source','lock','output','metadata','previous-run','checkpoint','commit']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--best-checkpoint',type=Path)
    p.add_argument('--worker',action='store_true')
    p.add_argument('--sample-stall-seconds',type=int,default=120)
    p.add_argument('--progress-stall-seconds',type=int,default=300)
    p.add_argument('--io-stall-seconds',type=int,default=900)
    a=p.parse_args()
    if a.worker:worker(a)
    else:
        guard=helper(Path(__file__).with_name('run_text2event_guarded_pilot.py'),'full_guard')
        common.__file__=__file__;common.supervise(a,guard.stalled)


if __name__=='__main__':main()
