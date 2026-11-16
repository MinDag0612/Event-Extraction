"""Bounded full-TRAIN pilot using the frozen VHE model and training recipe."""
import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def select_samples(rows, count, seed):
    indices = sorted(random.Random(seed).sample(range(len(rows)), min(count, len(rows))))
    return [rows[index] for index in indices]


def validate_source(source, lock):
    manifest = source / 'package-manifest.json'
    if digest(manifest) != lock['source_manifest_sha256']:
        raise ValueError('Frozen VHE manifest differs')
    for name, expected in json.loads(manifest.read_text()).items():
        if digest(source / name) != expected:
            raise ValueError('Frozen source differs: ' + name)
    config = json.loads((source / 'configs/text2event/baseline-mt5.json').read_text())
    if config != lock['config']:
        raise ValueError('Model configuration differs from VHE')
    return config


def validate_resume(state, signature, commit, selected, expected_selected, updates_per_epoch):
    """Require full optimizer/RNG state and the completed matching subset evaluation."""
    required = {'model', 'optimizer', 'scheduler', 'epoch', 'best', 'signature', 'python_rng', 'torch_rng', 'cuda_rng'}
    if not required <= state.keys() or not state['optimizer'].get('state'):
        raise ValueError('Resume requires full optimizer and RNG state')
    if state['signature'] != {'run': signature, 'stage': 'full'} or state['epoch'] != 1:
        raise ValueError('Resume source signature/stage/epoch differs')
    if state['scheduler']['last_epoch'] != updates_per_epoch or state['cuda_rng'] is None:
        raise ValueError('Resume scheduler/RNG differs')
    if commit.get('stage') != 'full' or commit.get('epoch') != 1 or commit.get('scheduler_update') != updates_per_epoch:
        raise ValueError('Evaluation commit does not match checkpoint')
    if commit.get('selection_scope') != 'fixed DEV subset' or selected != expected_selected:
        raise ValueError('Resume selection scope or sample IDs differ')
    score = commit['results']['dev']['scores']['AC']['f1']
    if commit['best_subset_dev'] != score or state['best'] != -1.:
        raise ValueError('Unexpected source selection state')
    for split in ['train', 'dev']:
        if commit['results'][split]['diagnostics']['samples'] != len(selected[split]):
            raise ValueError('Incomplete source evaluation')
    return score


def diagnostic_summary(issues):
    valid = [item for item in issues if item['status'] == 'ok']
    return {'samples': len(issues), 'invalid': len(issues) - len(valid),
            'valid_empty': sum(item.get('generated', '').replace(' ', '') == '<extra_id_0><extra_id_1>' for item in valid),
            'without_eos': sum(item.get('reason') == 'generation_limit_without_eos' for item in issues),
            'at_generation_limit': sum(item.get('generated_tokens', 0) >= 704 for item in issues)}


def worker(a):
    output = a.output
    output.mkdir(parents=True, exist_ok=True)
    def event(phase, **fields):
        value = {'phase': phase, 'time': time.time(), **fields}
        write(output / 'activity.json', value)
        print(json.dumps(value), flush=True)

    event('PREFLIGHT')
    source = a.source.resolve()
    lock = json.loads(a.lock.read_text())
    config = validate_source(source, lock)
    sys.path.insert(0, str(source))
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer
    from scripts.train_text2event import build_scheduler
    from src.baselines.text2event.training import load_components, BundleDataset, BatchCollator, train_epoch
    from src.baselines.text2event.checkpoint import save_checkpoint, save_best
    from src.baselines.text2event.curriculum import evaluate_substructure, leaf_set
    from src.baselines.text2event.convert import convert_substructure
    from src.baselines.text2event.evaluation import evaluate_model
    # Verify imports originate in the immutable runtime rather than the local checkout.
    import inspect
    for function in [build_scheduler, train_epoch, evaluate_model, evaluate_substructure]:
        if not Path(inspect.getfile(inspect.unwrap(function))).resolve().is_relative_to(source):
            raise ValueError('Import escaped frozen runtime')
    folder = source / 'bundle' / a.dataset
    tokenizer = AutoTokenizer.from_pretrained(source / 'bundle/tokenizer', local_files_only=True)
    golds = {split: list(map(json.loads, (folder / (split + '.gold.jsonl')).read_text().splitlines())) for split in ['train', 'dev']}
    sizes = {'BKEE': (9341, 4039), 'GENEVA': (1874, 766), 'PHEE': (2853, 949)}
    if tuple(len(golds[s]) for s in ['train', 'dev']) != sizes[a.dataset]:
        raise ValueError('Unexpected dataset sizes')
    selected = {split: select_samples(rows, a.eval_samples, lock['trainer']['seed']) for split, rows in golds.items()}
    write(output / 'selected-samples.json', {split: [row['id'] for row in rows] for split, rows in selected.items()})
    preflight = {}
    for stage in ['substructure', 'full']:
        for split in ['train', 'dev']:
            ds = BundleDataset(folder, split, tokenizer, config['max_source_length'], config['max_target_length'], stage, config[stage + '_source_prefix'])
            preflight[stage + '-' + split] = {'samples': len(ds), 'max_source': max(len(row['input_ids']) for row in ds.examples), 'max_target': max(len(row['labels']) for row in ds.examples)}
    write(output / 'preflight.json', preflight)
    signature = {'implementation': lock['implementation'], 'config': config, 'dataset': a.dataset, **lock['trainer'],
                 'files': {str(p.relative_to(source / 'bundle')): digest(p) for p in list(folder.glob('*')) + list((source / 'bundle/tokenizer').glob('*')) if p.is_file()},
                 'pilot': {'substructure_epochs': 5, 'full_epochs': 1 if a.resume_state else a.full_epochs, 'evaluation_samples_per_split': a.eval_samples,
                           'selection_scope': 'fixed DEV subset; not full DEV baseline', 'test_used': False}}
    if a.resume_state:
        previous = json.loads(a.resume_run.read_text())
        if signature != previous:
            raise ValueError('Resume run/config/data signature differs')
        write(output / 'resume-request.json', {'source_epoch': 1, 'target_full_epoch': a.full_epochs,
              'source_checkpoint_sha256': digest(a.resume_state), 'original_signature_preserved': True,
              'test_used': False, 'evaluation_samples_per_split': a.eval_samples})
    write(output / 'run.json', signature)
    if a.preflight_only:
        write(output / 'status.json', {'status': 'preflight_passed', 'trained': False, 'test_used': False})
        return
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    event('MODEL_LOAD')
    random.seed(signature['seed']); torch.manual_seed(signature['seed'])
    model, tokenizer = load_components(config, source / 'bundle/tokenizer')
    model.to('cuda'); model.gradient_checkpointing_enable(); model.config.use_cache = False
    write(output / 'model-loading.json', model.text2event_load_report)
    normalize = tokenizer.backend_tokenizer.normalizer.normalize_str
    results = {}
    stages = [('full', a.full_epochs, 30)] if a.resume_state else [('substructure', 5, 5), ('full', a.full_epochs, 30)]
    for stage, invocation_epochs, schedule_epochs in stages:
        stage_output = output / stage
        stage_output.mkdir(exist_ok=True)
        if stage == 'full' and not a.resume_state:
            event('STAGE_HANDOFF')
            state = torch.load(output / 'substructure/best.pt', map_location='cpu', weights_only=False)
            model.load_state_dict(state['model'])
            write(output / 'handoff.json', {'selected_substructure_epoch': state['epoch'], 'subset_dev_score': state['best'],
                                          'optimizer_reset': True, 'scheduler_total_full_epochs': 30})
            del state
            # The selected weights and all last-epoch diagnostics remain available.
            (output / 'substructure/pre-dev-state.pt').unlink()
        random.seed(signature['seed']); torch.manual_seed(signature['seed'])
        ds = BundleDataset(folder, 'train', tokenizer, config['max_source_length'], config['max_target_length'], stage, config[stage + '_source_prefix'])
        loader = DataLoader(ds, batch_size=signature['micro_batch'], shuffle=True, collate_fn=BatchCollator(tokenizer))
        optimizer = torch.optim.AdamW(model.parameters(), lr=signature['lr'], weight_decay=0.)
        scheduler, schedule = build_scheduler(optimizer, len(loader), signature['accumulation'], schedule_epochs,
                                             signature['scheduler'], signature['warmup_steps'], signature['warmup_ratio'])
        write(stage_output / 'schedule.json', schedule)
        stage_signature = {'run': signature, 'stage': stage}
        best = -1.
        start_epoch = 1
        if a.resume_state:
            event('CHECKPOINT_RESTORE')
            state = torch.load(a.resume_state, map_location='cpu', weights_only=False)
            commit = json.loads(a.resume_commit.read_text())
            original_selected = json.loads(a.resume_samples.read_text())
            expected_selected = {split: [row['id'] for row in rows] for split, rows in selected.items()}
            best = validate_resume(state, signature, commit, original_selected, expected_selected, schedule['total_optimizer_updates'] // 30)
            if len(state['cuda_rng']) != torch.cuda.device_count():
                raise ValueError('CUDA RNG device count differs from source')
            model.load_state_dict(state['model'])
            optimizer.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler'])
            random.setstate(state['python_rng']); torch.set_rng_state(state['torch_rng'])
            torch.cuda.set_rng_state_all(state['cuda_rng'])
            start_epoch = state['epoch'] + 1
            write(output / 'restore.json', {'restored_epoch': state['epoch'], 'scheduler_update': scheduler.last_epoch,
                  'best_subset_dev': best, 'optimizer_state_entries': len(optimizer.state),
                  'python_rng_restored': random.getstate() == state['python_rng'],
                  'cpu_rng_restored': torch.equal(torch.get_rng_state(), state['torch_rng']),
                  'cuda_rng_restored': all(torch.equal(x, y) for x, y in zip(torch.cuda.get_rng_state_all(), state['cuda_rng'])),
                  'train_order': 'restored global RNG; unchanged DataLoader and full TRAIN'})
            del state
            event('CHECKPOINT_SAVE', stage='full', epoch=1)
            save_best(stage_output / 'best.pt', model, 1, best, stage_signature)
            if a.remove_resume_copy:
                a.resume_state.unlink()
                write(output / 'storage-retention.json', {'temporary_download_removed_after_restore': True,
                      'cloud_source_preserved': True, 'source_epoch1_weights_retained_in': 'full/best.pt'})
            write(stage_output / 'epoch-1-evaluation-commit.json', commit)
            results['full-1'] = commit['results']
            event('RESTORED', stage='full', epoch=1, scheduler_update=scheduler.last_epoch, best_subset_dev=best)
        class Observer:
            def before_step(self, model, optimizer, gradient_norm):
                pass
            def after_step(self, model):
                event('TRAIN_PROGRESS', stage=stage, epoch=epoch, optimizer_update=scheduler.last_epoch)
        for epoch in range(start_epoch, invocation_epochs + 1):
            event('TRAIN_PROGRESS', stage=stage, epoch=epoch, optimizer_update=scheduler.last_epoch)
            began = time.monotonic()
            metrics = train_epoch(model, loader, optimizer, scheduler, 'cuda', signature['accumulation'], progress_every=64, step_observer=Observer())
            metrics['train_seconds'] = time.monotonic() - began
            write(stage_output / f'epoch-{epoch}-train.json', metrics)
            event('CHECKPOINT_SAVE', stage=stage, epoch=epoch)
            snapshot = stage_output / 'pre-dev-state.pt'
            save_checkpoint(snapshot, model, optimizer, scheduler, epoch, best, stage_signature)
            write(stage_output / 'snapshot.json', {'epoch': epoch, 'stage': stage, 'phase': 'before_evaluation',
                  'checkpoint': 'pre-dev-state.pt', 'previous_best_subset_dev': best, 'scheduler_update': scheduler.last_epoch,
                  'resume_protocol': 'restore pre-dev-state plus matching evaluation commit; not stock resume'})
            epoch_results = {}
            python_rng = random.getstate()
            with torch.random.fork_rng():
                for split in ['train', 'dev']:
                    issues = []; predictions = []; tp = fp = fn = 0
                    began = time.monotonic()
                    with (stage_output / f'epoch-{epoch}-{split}.jsonl').open('w') as stream:
                        for index, sample in enumerate(selected[split], 1):
                            event('SAMPLE_START', stage=stage, epoch=epoch, split=split, index=index, id=sample['id'])
                            if stage == 'substructure':
                                _, one = evaluate_substructure(model, tokenizer, [sample], folder / 'event.schema', config, 'cuda')
                                issue = one[0]
                                predicted = leaf_set(issue['generated'], normalize) if issue['status'] == 'ok' else set()
                                gold = leaf_set(convert_substructure(sample)['event'], normalize)
                                tp += len(predicted & gold); fp += len(predicted - gold); fn += len(gold - predicted)
                            else:
                                _, pred, one = evaluate_model(model, tokenizer, [sample], folder / 'event.schema', config, 'cuda')
                                predictions.extend(pred); issue = one[0]
                            issues.append(issue)
                            stream.write(json.dumps(issue, ensure_ascii=False) + '\n'); stream.flush()
                            event('SAMPLE_DONE', stage=stage, epoch=epoch, split=split, index=index, status=issue['status'])
                    if stage == 'substructure':
                        scores = {'precision': tp / (tp + fp) if tp + fp else 0., 'recall': tp / (tp + fn) if tp + fn else 0.,
                                  'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.}
                    else:
                        from src.baselines.text2event.evaluation import score_by_id
                        scores = score_by_id(predictions, selected[split])
                        write(stage_output / f'epoch-{epoch}-{split}-predictions.json', predictions)
                    epoch_results[split] = {'scores': scores, 'diagnostics': diagnostic_summary(issues), 'seconds': time.monotonic() - began}
            random.setstate(python_rng)
            score = epoch_results['dev']['scores']['f1'] if stage == 'substructure' else epoch_results['dev']['scores']['AC']['f1']
            if score > best:
                best = score
                event('CHECKPOINT_SAVE', stage=stage, epoch=epoch)
                save_best(stage_output / 'best.pt', model, epoch, best, stage_signature)
            write(stage_output / f'epoch-{epoch}-evaluation-commit.json', {'epoch': epoch, 'stage': stage, 'best_subset_dev': best,
                 'selection_scope': 'fixed DEV subset', 'results': epoch_results, 'scheduler_update': scheduler.last_epoch})
            results[stage + '-' + str(epoch)] = epoch_results
            write(output / 'results.json', results)
            print(json.dumps({'stage': stage, 'epoch': epoch, 'results': epoch_results}), flush=True)
        del optimizer, scheduler, loader, ds
        torch.cuda.empty_cache()
    write(output / 'status.json', {'status': 'short_pilot_complete', 'baseline_complete': False, 'trained': True, 'test_used': False,
         'dataset': a.dataset, 'substructure_epochs': 5, 'full_epochs': a.full_epochs, 'evaluation_samples_per_split': a.eval_samples,
         'selection_scope': 'fixed DEV subset', 'automatic_continuation': False})


def supervise(a, stalled):
    import os
    import signal
    import subprocess
    import threading
    a.output.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, '-u', str(Path(__file__).resolve()), *sys.argv[1:], '--worker']
    with (a.output / 'pilot.log').open('w') as log:
        child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
        def pump():
            for line in child.stdout:
                print(line, end='', flush=True); log.write(line); log.flush()
        thread = threading.Thread(target=pump, daemon=True); thread.start()
        began = time.time()
        while child.poll() is None:
            activity = a.output / 'activity.json'
            event = json.loads(activity.read_text()) if activity.exists() else {'phase': 'BOOTSTRAP', 'time': began}
            if stalled(event, time.time(), a.sample_stall_seconds, a.progress_stall_seconds, a.io_stall_seconds):
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL); child.wait()
                write(a.output / 'status.json', {'status': 'pilot_interrupted', 'reason': 'phase_progress_watchdog',
                      'last_activity': event, 'baseline_complete': False, 'test_used': False,
                      'snapshots': [str(p.relative_to(a.output)) for p in a.output.glob('*/pre-dev-state.pt')]})
                break
            time.sleep(1)
        thread.join(timeout=5)
    if not (a.output / 'status.json').exists():
        write(a.output / 'status.json', {'status': 'failed', 'returncode': child.returncode, 'baseline_complete': False, 'test_used': False})
    status = json.loads((a.output / 'status.json').read_text())
    print('SHORT_PILOT_STATUS', json.dumps(status), flush=True)
    if status['status'] == 'failed':
        raise RuntimeError('Pilot worker failed; inspect pilot.log')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['source', 'lock', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--dataset', choices=['BKEE', 'GENEVA', 'PHEE'], required=True)
    parser.add_argument('--full-epochs', type=int, choices=[1, 3], default=1)
    parser.add_argument('--eval-samples', type=int, default=32)
    parser.add_argument('--sample-stall-seconds', type=int, default=120)
    parser.add_argument('--progress-stall-seconds', type=int, default=300)
    parser.add_argument('--io-stall-seconds', type=int, default=900)
    for name in ['resume-state', 'resume-run', 'resume-commit', 'resume-samples']:
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--remove-resume-copy', action='store_true')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--preflight-only', action='store_true')
    a = parser.parse_args()
    resume_fields = [a.resume_state, a.resume_run, a.resume_commit, a.resume_samples]
    if any(resume_fields) and not all(resume_fields):
        parser.error('Resume requires state/run/commit/sample IDs together')
    if a.resume_state and (a.dataset not in {'PHEE', 'BKEE'} or a.full_epochs != 3):
        parser.error('This continuation is bounded to PHEE/BKEE full epochs2–3')
    if a.remove_resume_copy and not a.resume_state:
        parser.error('No temporary resume download specified')
    if min(a.eval_samples, a.sample_stall_seconds, a.progress_stall_seconds, a.io_stall_seconds) < 1:
        parser.error('Counts and watchdog limits must be positive')
    if a.worker or a.preflight_only:
        worker(a)
    else:
        # The supervisor only observes phase progress; bounded sample sets replace full-DEV evaluation.
        import importlib.util
        spec = importlib.util.spec_from_file_location('pilot_supervisor', Path(__file__).with_name('run_text2event_guarded_pilot.py'))
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        supervise(a, module.stalled)


if __name__ == '__main__':
    main()
