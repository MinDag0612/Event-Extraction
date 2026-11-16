"""Validate legacy runtime and execute the pinned author's two-stage CLI."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time


def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2))


def verify_sources(author,lock):
    """Verify all tracked author files without patching their implementation."""
    for name,digest in lock['files'].items():
        if hashlib.sha256((author/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Modified author file: '+name)


def audit_data(author,data,model_path,output):
    """Audit slow-tokenizer targets and native constraints before invoking training."""
    import torch
    import transformers
    import datasets
    from transformers import AutoTokenizer
    sys.path.insert(0,str(author))
    from extraction.event_schema import EventSchema
    from extraction.extract_constraint import get_constraint_decoder
    from extraction.extraction_metrics import get_extract_metrics
    env={'python':sys.version,'torch':torch.__version__,'transformers':transformers.__version__,
         'datasets':datasets.__version__,'cuda':torch.cuda.is_available(),'visible_gpu_count':torch.cuda.device_count()}
    write(output/'environment.json',env)
    if sys.version_info[:2]!=(3,8) or not torch.__version__.startswith('1.7.1') or transformers.__version__!='4.4.2':
        raise RuntimeError('Legacy environment does not match the pinned author runtime')
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    tokenizer=AutoTokenizer.from_pretrained(str(model_path),use_fast=False)
    if tokenizer.encode('<extra_id_0> <extra_id_1>')!=[32099,32098,1]:
        tokenizer.add_special_tokens({'additional_special_tokens':['<extra_id_0>','<extra_id_1>']})
    if tokenizer.is_fast:raise ValueError('Author decoding requires the slow tokenizer')
    result={}
    for stage,suffix,format_name,prefix in [('span','_span','treespan','span: '),('full','','tree','event: ')]:
        folder=data/('GENEVA'+suffix)
        schema=EventSchema.read_from_file(str(folder/'event.schema'))
        sequences={name:tuple(tokenizer.encode(name,add_special_tokens=False)) for name in schema.type_list+schema.role_list}
        conflicts=[(a,b) for a,x in sequences.items() for b,y in sequences.items() if a!=b and len(x)<=len(y) and y[:len(x)]==x]
        if conflicts:raise ValueError('Native label trie has prefix collisions: '+str(conflicts[:5]))
        source_max=0;target_max=0
        for split in ['train','val']:
            for row in map(json.loads,(folder/(split+'.json')).read_text().splitlines()):
                source_max=max(source_max,len(tokenizer.encode(prefix+row['text'])))
                target_max=max(target_max,len(tokenizer.encode(row['event'])))
        checked=0
        smoke=data/('GENEVA-smoke'+suffix)
        for split in ['train','val']:
            for row in map(json.loads,(smoke/(split+'.json')).read_text().splitlines()):
                constraint=get_constraint_decoder(tokenizer,schema,format_name,source_prefix=prefix)
                source=tokenizer.encode(prefix+row['text']);generated=[tokenizer.pad_token_id]
                for position,token in enumerate(tokenizer.encode(row['event'])):
                    allowed=constraint.constraint_decoding(torch.tensor(source),torch.tensor(generated))
                    if token not in allowed:raise ValueError('Native decoder masks gold at '+stage+'/'+split+'/'+str(position))
                    generated.append(token)
                checked+=1
        gold=[r['event'] for r in map(json.loads,(smoke/'val.json').read_text().splitlines())]
        self_score=get_extract_metrics(gold,gold,schema,format_name)
        result[stage]={'source_max':source_max,'target_max':target_max,
            'source_length':max(256,int(math.ceil(source_max/32.))*32),
            'target_length':max(128,int(math.ceil((target_max+1)/32.))*32),
            'native_gold_prefix_checked':checked,'label_prefix_conflicts':0,'gold_self_score':self_score}
        print(json.dumps({'phase':'AUTHOR_DATA_AUDIT','stage':stage,**result[stage]}),flush=True)
    write(output/'data-audit.json',result)
    return result


def command(author,data,model_path,output,stage,lengths,smoke,small=False):
    """Build native CLI arguments from the author's curriculum bash defaults."""
    span=stage=='span';folder=data/(('GENEVA-smoke' if smoke or small else 'GENEVA')+('_span' if span else ''))
    steps=2 if smoke else (8 if small else 500)
    args=[sys.executable,str(author/'run_seq2seq.py'),'--do_train','--do_eval','--constraint_decoding',
        '--predict_with_generate','--use_fast_tokenizer','False','--label_smoothing_sum','False',
        '--evaluation_strategy','steps','--eval_steps',str(steps),'--save_steps',str(steps),'--logging_steps','1',
        '--save_total_limit','1','--load_best_model_at_end','--metric_for_best_model','eval_trigger-F1' if span else 'eval_role-F1',
        '--max_source_length',str(lengths['source_length']),'--max_target_length',str(lengths['target_length']),
        '--val_max_target_length',str(lengths['target_length']),'--num_beams','1',
        '--num_train_epochs','5' if span else '30','--task','event',
        '--train_file',str(folder/'train.json'),'--validation_file',str(folder/'val.json'),
        '--event_schema',str(folder/'event.schema'),'--per_device_train_batch_size','1',
        '--gradient_accumulation_steps','1' if smoke or small else '16','--per_device_eval_batch_size','1',
        '--output_dir',str(output),'--model_name_or_path',str(model_path),'--learning_rate','5e-5',
        '--lr_scheduler_type','constant_with_warmup','--warmup_steps','0','--label_smoothing_factor','0',
        '--decoding_format','treespan' if span else 'tree','--source_prefix','span: ' if span else 'event: ',
        '--seed','421','--disable_tqdm','True']
    if smoke:args+=['--max_steps','2']
    return args


def evaluation_command(author,data,output,stage,lengths,split):
    """Evaluate the selected native checkpoint on fixed TRAIN or DEV examples."""
    folder=data/('GENEVA-smoke'+('_span' if stage=='span' else ''))
    return [sys.executable,str(author/'run_seq2seq.py'),'--do_eval','--constraint_decoding',
        '--predict_with_generate','--use_fast_tokenizer','False','--task','event',
        '--model_name_or_path',str(output/stage),'--output_dir',str(output/(stage+'-'+split+'-evaluation')),
        '--validation_file',str(folder/('train.json' if split=='train' else 'val.json')),
        '--event_schema',str(folder/'event.schema'),'--per_device_eval_batch_size','1',
        '--max_source_length',str(lengths['source_length']),'--max_target_length',str(lengths['target_length']),
        '--val_max_target_length',str(lengths['target_length']),'--num_beams','1',
        '--decoding_format','treespan' if stage=='span' else 'tree',
        '--source_prefix','span: ' if stage=='span' else 'event: ','--seed','421','--disable_tqdm','True']


def prune_completed_optimizer_states(stage_output):
    """Remove disk copies of optimizer states after native checkpoint writes finish."""
    removed=[]
    for checkpoint in stage_output.glob('checkpoint-*'):
        marker=checkpoint/'trainer_state.json'
        try:
            state=json.loads(marker.read_text())
            if state['global_step']!=int(checkpoint.name.split('-')[-1]):continue
            if not (checkpoint/'pytorch_model.bin').is_file():continue
            for name in ['optimizer.pt','scheduler.pt']:
                path=checkpoint/name
                if path.is_file():
                    size=path.stat().st_size;path.unlink();removed.append({'file':str(path),'bytes':size})
        except (FileNotFoundError,json.JSONDecodeError,KeyError):
            continue
    return removed


def execute_native(args,author,log_path,prune_output=None):
    """Stream native process output to the notebook and a persistent stage log."""
    with log_path.open('w') as log:
        p=subprocess.Popen(args,cwd=author,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True,bufsize=1)
        for line in p.stdout:
            print(line,end='',flush=True);log.write(line);log.flush()
            if prune_output is not None and 'Saving model checkpoint to ' in line:
                removed=prune_completed_optimizer_states(prune_output)
                if removed:
                    message=json.dumps({'phase':'COMPLETED_OPTIMIZER_STATE_PRUNED','files':removed})+'\n'
                    print(message,end='',flush=True);log.write(message);log.flush()
        result=p.wait()
        if prune_output is not None:prune_completed_optimizer_states(prune_output)
        return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['author','data','model','output','lock']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--mode',choices=['smoke','small','baseline'],default='smoke')
    a=parser.parse_args();a.author=a.author.resolve();a.data=a.data.resolve();a.model=a.model.resolve();a.output=a.output.resolve()
    a.output.mkdir(parents=True,exist_ok=False)
    lock=json.loads(a.lock.read_text());verify_sources(a.author,lock)
    audit=audit_data(a.author,a.data,a.model,a.output)
    smoke=a.mode=='smoke';small=a.mode=='small'
    planned={stage:command(a.author,a.data,a.model if stage=='span' else a.output/'span',a.output/stage,stage,audit[stage],smoke,small)
             for stage in ['span','full']}
    write(a.output/'run.json',{'mode':a.mode,'author_commit':lock['commit'],'commands':planned,
        'test_used':False,'checkpoint_resume_supported':not (smoke or small),'optimizer_recipe':'author curriculum bash defaults','label_mapping':str(a.data/'label-map.json')})
    for stage in ['span','full']:
        if shutil.disk_usage(a.output).free<10*1024**3:raise OSError('Need10GiB for native model and optimizer checkpoint')
        started=time.time();print(json.dumps({'phase':'AUTHOR_TRAIN_START','stage':stage,'command':planned[stage]}),flush=True)
        result=execute_native(planned[stage],a.author,a.output/(stage+'.log'),
                              a.output/stage if smoke or small else None)
        write(a.output/(stage+'-status.json'),{'returncode':result,'seconds':time.time()-started})
        if result:raise RuntimeError('Native author CLI failed: '+stage)
        if not (a.output/stage/'pytorch_model.bin').exists():raise RuntimeError('Native model was not saved')
        if smoke or small:
            removed=[]
            for path in (a.output/stage).glob('checkpoint-*'):
                shutil.rmtree(path);removed.append(path.name)
            write(a.output/(stage+'-retention.json'),{'root_model_retained':True,'completed_smoke_optimizer_checkpoints_removed':removed})
        if small:
            for split in ['train','dev']:
                args=evaluation_command(a.author,a.data,a.output,stage,audit[stage],split)
                print(json.dumps({'phase':'AUTHOR_SELECTED_MODEL_EVALUATION','stage':stage,'split':split,'command':args}),flush=True)
                result=execute_native(args,a.author,a.output/(stage+'-'+split+'-evaluation.log'))
                if result:raise RuntimeError('Native evaluation failed: '+stage+'/'+split)
    verify_sources(a.author,lock)
    write(a.output/'completion.json',{'status':'complete','mode':a.mode,'native_stages':['span','full'],
        'author_source_unchanged':True,'test_used':False,'curriculum_training_complete':not smoke,'baseline_complete':False})


if __name__=='__main__':main()
