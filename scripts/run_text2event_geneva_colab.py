"""Portable GENEVA checkpoint diagnosis and guarded one-epoch continuation."""
import argparse
import inspect
import json
import random
import signal
import sys
import time
from pathlib import Path


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temporary.replace(path)


def guard_reason(issues, elapsed, *, max_seconds=600):
    if elapsed >= max_seconds:return 'evaluation_time_budget'
    if len(issues)>=32 and all(x.get('generated')=='<extra_id_0><extra_id_1>' for x in issues[:32]):return 'first_32_predictions_all_empty'
    if len(issues)>=3 and all(x.get('generated_tokens',0)>=704 for x in issues[-3:]):return 'three_consecutive_generation_limits'
    if len(issues)>=16 and sum(x.get('status')!='ok' for x in issues)/len(issues)>.25:return 'invalid_rate_above_25_percent'
    return None


class GenerationGuard(RuntimeError):
    pass


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--mode',choices=['diagnose','train-one'],default='diagnose')
    p.add_argument('--compare-bounded',action='store_true');a=p.parse_args()
    import torch
    sys.path.insert(0,str(a.source))
    from src.baselines.text2event.training import load_components
    from src.baselines.text2event.curriculum import evaluate_substructure,leaf_set
    from src.baselines.text2event.convert import convert_substructure
    from src.baselines.text2event.evaluation import evaluate_model,score_by_id
    import scripts.train_text2event as trainer
    from src.baselines.text2event.checkpoint import save_checkpoint
    folder=a.source/'bundle/GENEVA';config=json.loads((a.source/'configs/text2event/baseline-mt5.json').read_text())
    rows=list(map(json.loads,(folder/'dev.gold.jsonl').read_text().splitlines()));assert len(rows)==766
    indices=random.Random(421).sample(range(len(rows)),32);probe=[rows[i] for i in indices]
    write(a.output/'probe-ids.json',[r['id'] for r in probe])
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=False,mmap=True)
    run=state['signature']['run'];stage=state['signature']['stage']
    if run['dataset']!='GENEVA' or run['config']!=config or stage not in {'substructure','full'}:raise ValueError('Checkpoint signature differs')
    if not torch.cuda.is_available():raise RuntimeError('Choose a GPU runtime first')

    def alarm_handler(*_):raise GenerationGuard('sample_exceeded_120_seconds')
    signal.signal(signal.SIGALRM,alarm_handler)

    def evaluate_guarded(model,tokenizer,samples,schema,cfg,device, *, flat, label):
        issues=[];predictions=[];tp=fp=fn=0;began=time.monotonic();normalize=tokenizer.backend_tokenizer.normalizer.normalize_str
        dest=a.output/'guard-logs'/f'{label}.jsonl';dest.parent.mkdir(parents=True,exist_ok=True)
        def stop(reason):
            write(a.output/'guard-status.json',{'status':'diagnostic_stopped','reason':reason,'label':label,'evaluated_samples':len(issues),
                'requested_samples':len(samples),'partial':True,'dev_complete':False,'test_evaluated':False})
            raise GenerationGuard(reason)
        with dest.open('w') as stream:
            for index,row in enumerate(samples,1):
                try:
                    signal.alarm(120)
                    if flat:_,one=evaluate_substructure(model,tokenizer,[row],schema,cfg,device);pred=None
                    else:_,pred,one=evaluate_model(model,tokenizer,[row],schema,cfg,device)
                except GenerationGuard as e:stop(str(e))
                finally:signal.alarm(0)
                issue=one[0];issues.append(issue)
                if flat:
                    predicted=leaf_set(issue['generated'],normalize) if issue['status']=='ok' else set()
                    gold=leaf_set(convert_substructure(row)['event'],normalize)
                    tp+=len(predicted & gold);fp+=len(predicted-gold);fn+=len(gold-predicted)
                else:predictions.extend(pred)
                stream.write(json.dumps({'diagnostic':issue,'prediction':pred[0] if pred else None},ensure_ascii=False)+'\n');stream.flush()
                print(json.dumps({'phase':'DEV_SAMPLE','label':label,'index':index,'total':len(samples),'status':issue['status'],'tokens':issue.get('generated_tokens')}),flush=True)
                reason=guard_reason(issues,time.monotonic()-began,max_seconds=600 if len(samples)<=32 else 3600)
                if reason:stop(reason)
            scores={'precision':tp/(tp+fp) if tp+fp else 0.,'recall':tp/(tp+fn) if tp+fn else 0.,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.} if flat else score_by_id(predictions,samples)
            # Empty-root collapse is also a diagnostic failure; no fake full-DEV score.
            if len(samples)>=32 and all(x.get('generated')=='<extra_id_0><extra_id_1>' for x in issues[:32]):stop('first_32_predictions_all_empty')
            write(a.output/'guard-logs'/f'{label}-results.json',{'samples':len(samples),'scores':scores,'seconds':time.monotonic()-began,'partial':False})
            return (scores,issues) if flat else (scores,predictions,issues)

    if a.mode=='diagnose':
        model,tokenizer=load_components(config,a.source/'bundle/tokenizer');model.load_state_dict(state['model']);model.to('cuda');del state
        variants=[('upstream-compatible',config)]
        if a.compare_bounded:
            bounded=json.loads(json.dumps(config));bounded['substructure_constraint_decoder']='project-constrained';bounded['constraint_decoder']='project-constrained'
            bounded['constraint_repetition_limits']['GENEVA']['flat_repeat_limit']=1
            variants.append(('bounded-experiment',bounded))
        results={}
        for label,cfg in variants:
            try:
                value=evaluate_guarded(model,tokenizer,probe,folder/'event.schema',cfg,'cuda',flat=stage=='substructure',label=label)
                results[label]={'status':'probe_complete','scores':value[0],'samples':32}
            except GenerationGuard as e:results[label]={'status':'diagnostic_stopped','reason':str(e),'partial':True}
        write(a.output/'diagnosis.json',{'stage':stage,'source_checkpoint_epoch':torch.load(a.checkpoint,map_location='cpu',weights_only=False,mmap=True)['epoch'],
             'scope':'fixed DEV32 diagnosis; not checkpoint selection, not full DEV or TEST','variants':results})
        print(json.dumps(results),flush=True);return

    # Native signature stays unchanged. CUDA streams are explicitly mapped for 2→1 GPU portability.
    if not (a.output/'run.json').exists():raise ValueError('Restore complete Kaggle run metadata before train-one')
    def restore(path,model,optimizer,scheduler,signature,early_stopping=None):
        s=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
        if s['signature']!=signature:raise ValueError('Resume signature differs')
        if early_stopping is not None:early_stopping.load_state_dict(s['early_stopping'])
        model.load_state_dict(s['model']);optimizer.load_state_dict(s['optimizer']);scheduler.load_state_dict(s['scheduler'])
        random.setstate(s['python_rng']);torch.set_rng_state(s['torch_rng'])
        rng=s['cuda_rng'];count=torch.cuda.device_count()
        if len(rng)<count:raise ValueError('Cannot invent additional GPU RNG states')
        torch.cuda.set_rng_state_all(rng[:count])
        write(a.output/'colab-portability.json',{'source_cuda_rng_states':len(rng),'target_devices':count,'bitwise_equivalence_claimed':False,
              'optimizer_scheduler_restored':True,'completed_epoch':s['epoch']})
        return s['epoch'],s['best']
    trainer.restore_checkpoint=restore;del state

    def guarded(flat):
        def wrapped(model,tokenizer,samples,schema,cfg,device):
            frame=inspect.currentframe().f_back;local=frame.f_locals;current_epoch=local['epoch']+1
            current_stage=local['stage'];stage_output=local['output'];pending=stage_output/'pending-dev.pt'
            save_checkpoint(pending,model,local['optimizer'],local['scheduler'],current_epoch,local['best'],local['stage_signature'],early_stopping=local['stopper'])
            write(stage_output/'pending-dev.json',{'epoch':current_epoch,'phase':'after_training_before_DEV','resume_with_stock_trainer':False})
            result=evaluate_guarded(model,tokenizer,samples,schema,cfg,device,flat=flat,label=f'{current_stage}-{current_epoch}')
            # Stock trainer commits last.pt/metrics only after this successful return.
            return result
        return wrapped
    trainer.evaluate_substructure=guarded(True);trainer.evaluate_model=guarded(False)
    sys.argv=['train_text2event','--bundle',str(a.source/'bundle'),'--dataset','GENEVA','--config',str(a.source/'configs/text2event/baseline-mt5.json'),
        '--output',str(a.output),'--epochs','30','--substructure-epochs','5','--micro-batch','1','--accumulation','16',
        '--lr','0.0001','--lr-scheduler','linear','--warmup-steps','0','--warmup-ratio','0.05','--seed','421',
        '--patience','3','--min-delta','0.001','--min-full-epochs','5','--gradient-checkpointing','--max-epochs-this-run','1',
        '--prune-completed-state','--progress-every','64','--resume']
    try:
        trainer.main()
        for pending in a.output.glob('*/pending-dev.pt'):pending.unlink()
        for pending in a.output.glob('*/pending-dev.json'):pending.unlink()
    except GenerationGuard as e:
        print('DIAGNOSTIC_STOP',str(e),'Committed last.pt is unchanged; pending-dev.pt retained; no TEST.',flush=True)
        raise SystemExit(2)


if __name__=='__main__':main()
