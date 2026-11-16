"""Run isolated GENEVA preflight or start its pinned training recipe."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from src.baselines.text2event_clean.preflight import run, bundle_hashes


def validate_recipe(config, recipe):
    """Require upstream-compatible decoding and a fixed multilingual training flow."""
    if config['model_name']!='google/mt5-base':raise ValueError('Expected mT5-base')
    if any(config.get(key)!='upstream-compatible' for key in ['constraint_decoder','substructure_constraint_decoder']):
        raise ValueError('Expected upstream-compatible decoder')
    if config.get('empty_root_penalty',0)!=0 or not config.get('constraint_allow_empty_tree',True):
        raise ValueError('Unexpected root override')
    if recipe['dataset']!='GENEVA' or (recipe['substructure_epochs'],recipe['full_epochs'])!=(5,30):
        raise ValueError('Unexpected dataset/curriculum')


def training_command(bundle, config_path, recipe, output, invocation_epochs):
    """Build an explicit fresh-training command with independent stage schedules."""
    return [sys.executable,'-u','-m','scripts.train_text2event','--bundle',str(bundle),
        '--dataset','GENEVA','--config',str(config_path),'--output',str(output),
        '--substructure-epochs',str(recipe['substructure_epochs']),'--epochs',str(recipe['full_epochs']),
        '--micro-batch',str(recipe['micro_batch']),'--accumulation',str(recipe['accumulation']),
        '--lr',str(recipe['lr']),'--lr-scheduler',recipe['scheduler'],'--warmup-ratio',str(recipe['warmup_ratio']),
        '--patience','0','--seed',str(recipe['seed']),'--gradient-checkpointing',
        '--evaluation-progress','--progress-every','64','--max-epochs-this-run',str(invocation_epochs)]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--config',type=Path,default=Path('configs/text2event_clean/geneva-model.json'))
    p.add_argument('--recipe',type=Path,default=Path('configs/text2event_clean/geneva-recipe.json'))
    p.add_argument('--phase',choices=['preflight','pilot','train','checkpoint-audit','lr-branch','decoder-replay'],default='preflight')
    p.add_argument('--preflight-result',type=Path)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--branch',choices=['linear','constant'])
    p.add_argument('--consume-source',action='store_true')
    p.add_argument('--expected-checkpoint-epoch',type=int,default=1)
    p.add_argument('--invocation-epochs',type=int,default=1)
    a=p.parse_args();config=json.loads(a.config.read_text());recipe=json.loads(a.recipe.read_text())
    validate_recipe(config,recipe)
    if a.phase=='decoder-replay':
        if a.checkpoint is None:raise ValueError('Checkpoint required')
        from src.baselines.text2event_clean.decoder_replay import run as replay_decoders
        replay_decoders(a.bundle,config,a.checkpoint,a.output,a.expected_checkpoint_epoch)
    elif a.phase=='lr-branch':
        if a.checkpoint is None or a.branch is None:raise ValueError('Branch and full-state checkpoint required')
        from src.baselines.text2event_clean.lr_ablation import run as run_branch
        run_branch(a.bundle,config,recipe,a.checkpoint,a.output,a.branch,a.consume_source)
    elif a.phase=='checkpoint-audit':
        if a.checkpoint is None:raise ValueError('Checkpoint required')
        from src.baselines.text2event_clean.checkpoint_audit import run as audit_checkpoint
        audit_checkpoint(a.bundle,config,a.checkpoint,a.output,a.expected_checkpoint_epoch)
    elif a.phase=='preflight':run(a.bundle,config,recipe,a.output)
    else:
        if a.preflight_result is None:raise ValueError('Training requires a passed matching preflight')
        completion=json.loads((a.preflight_result/'completion.json').read_text())
        request=json.loads((a.preflight_result/'request.json').read_text())
        if (completion['status']!='passed' or request['config']!=config or request['recipe']!=recipe
                or request['bundle_hashes']!=bundle_hashes(a.bundle)):
            raise ValueError('Preflight does not match training recipe')
        if a.output.exists() or a.invocation_epochs<1:raise ValueError('Expected new output and positive epoch budget')
        if a.phase=='pilot':
            from src.baselines.text2event_clean.pilot import run as run_pilot
            run_pilot(a.bundle,config,recipe,a.output)
            return
        subprocess.run(training_command(a.bundle,a.config,recipe,a.output,a.invocation_epochs),check=True)


if __name__=='__main__':main()
