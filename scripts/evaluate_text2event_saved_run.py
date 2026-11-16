"""Evaluate a completed VHE checkpoint with its original packaged implementation."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time


def digest(path):
    """Hash an artifact without loading its full contents into memory."""
    value=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(8*1024*1024),b''):value.update(block)
    return value.hexdigest()


def write(path,value):
    """Write a completed JSON artifact atomically."""
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    temporary.replace(path)


def validate_checkpoint(state,run,expected):
    """Check checkpoint provenance against the saved run and DEV selection."""
    if state.get('signature')!={'run':run,'stage':'full'}:raise ValueError('Checkpoint/run signature mismatch')
    if state.get('epoch')!=23 or expected.get('epoch')!=23:raise ValueError('Expected DEV-selected epoch23')
    if abs(state['best']-expected['dev']['AC']['f1'])>1e-12:raise ValueError('Checkpoint selection score mismatch')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--manifest-sha256',required=True)
    args=parser.parse_args();source=args.source.resolve();run_path=args.run.resolve();output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    manifest_path=source/'package-manifest.json'
    if digest(manifest_path)!=args.manifest_sha256:raise ValueError('Unexpected source manifest')
    manifest=json.loads(manifest_path.read_text())
    for name,expected in manifest.items():
        if digest(source/name)!=expected:raise ValueError('Original source/data changed: '+name)
    run=json.loads((run_path/'run.json').read_text())
    if run['dataset']!='VHE':raise ValueError('Expected VHE')
    for name,expected in run['files'].items():
        if digest(source/'bundle'/name)!=expected:raise ValueError('Run bundle mismatch: '+name)
    for stage,epochs in [('substructure',5),('full',30)]:
        completion=json.loads((run_path/stage/'completion.json').read_text())
        if completion['epochs_completed']!=epochs:raise ValueError('Training stage not complete: '+stage)
    sys.path.insert(0,str(source))
    import torch
    import transformers
    from src.baselines.text2event.training import load_components
    from src.baselines.text2event.evaluation import evaluate_model,score_by_id
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    checkpoint=run_path/'full/best.pt';checkpoint_sha=digest(checkpoint)
    state=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
    expected=json.loads((run_path/'full/epoch-23-metrics.json').read_text())
    validate_checkpoint(state,run,expected)
    config=run['config'];model,tokenizer=load_components(config,source/'bundle/tokenizer')
    model.load_state_dict(state['model'],strict=True);del state
    model.to('cuda');model.eval()
    write(output/'preflight.json',{'checkpoint_epoch':23,'checkpoint_sha256':checkpoint_sha,
        'source_manifest_sha256':args.manifest_sha256,'implementation':run['implementation'],
        'torch':torch.__version__,'transformers':transformers.__version__,'training_performed':False,
        'checkpoint_selection':'saved DEV AC best, unchanged','config':config})
    schema=source/'bundle/VHE/event.schema'
    for split,count in [('dev',406),('test',397)]:
        golds=list(map(json.loads,(source/'bundle/VHE'/(split+'.gold.jsonl')).read_text().splitlines()))
        if len(golds)!=count:raise ValueError('Unexpected split size: '+split)
        predictions=[];diagnostics=[];started=time.monotonic()
        with (output/(split+'-samples.jsonl')).open('w') as handle:
            for index,gold in enumerate(golds,1):
                print(json.dumps({'phase':'SAMPLE_START','split':split,'index':index,'total':count,'id':gold['id']}),flush=True)
                _,pred,issues=evaluate_model(model,tokenizer,[gold],schema,config,'cuda')
                predictions.extend(pred);diagnostics.extend(issues)
                handle.write(json.dumps({'prediction':pred[0],'diagnostic':issues[0]},ensure_ascii=False)+'\n');handle.flush()
                print(json.dumps({'phase':'SAMPLE_DONE','split':split,'index':index,'total':count,
                    'status':issues[0]['status'],'elapsed_seconds':time.monotonic()-started}),flush=True)
        scores=score_by_id(predictions,golds)
        write(output/(split+'-results.json'),{'checkpoint_epoch':23,'scores':scores,'diagnostics':diagnostics,
            'seconds':time.monotonic()-started,'samples':count,'checkpoint_sha256':checkpoint_sha})
        write(output/(split+'-predictions.json'),predictions)
        print(json.dumps({'phase':'SPLIT_COMPLETE','split':split,'scores':scores}),flush=True)
        if split=='dev':
            differences={task:{metric:scores[task][metric]-value for metric,value in expected['dev'][task].items()}
                         for task in expected['dev']}
            passed=all(abs(value)<1e-8 for task in differences.values() for value in task.values())
            write(output/'dev-reproduction.json',{'passed':passed,'differences':differences,'expected':expected['dev'],'actual':scores})
            if not passed:raise ValueError('DEV reproduction mismatch; TEST not evaluated')
    write(output/'completion.json',{'status':'complete','dataset':'VHE','mode':'evaluation-only',
        'checkpoint_epoch':23,'training_performed':False,'dev_verified':True,'test_evaluated':True,
        'test_used_for_selection':False,'checkpoint_sha256':checkpoint_sha})


if __name__=='__main__':main()
