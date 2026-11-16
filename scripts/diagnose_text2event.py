"""Bounded TRAIN-only memorization diagnostic; never a baseline result."""
import argparse
import json
import random
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from src.baselines.text2event.training import load_components, BundleDataset, BatchCollator, train_epoch
from src.baselines.text2event.evaluation import evaluate_model
from src.baselines.text2event.diagnostics import generation_summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--dataset', choices=['BKEE', 'GENEVA', 'PHEE', 'VHE'], required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--config', type=Path, default=Path('configs/text2event/preparation.json'))
    p.add_argument('--examples', type=int, default=8)
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--eval-every', type=int, default=5)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    if min(a.examples, a.epochs, a.eval_every) < 1 or a.lr <= 0:
        p.error('Require positive settings')
    a.output.mkdir(parents=True, exist_ok=False)
    random.seed(42); torch.manual_seed(42)
    config = json.loads(a.config.read_text())
    folder = a.bundle/a.dataset
    gold = list(map(json.loads, (folder/'train.gold.jsonl').read_text().splitlines()))
    # Short positives first; BKEE includes negatives without letting them dominate.
    positives = sorted([g for g in gold if g['events']], key=lambda g: len(g['tokens']))
    negatives = sorted([g for g in gold if not g['events']], key=lambda g: len(g['tokens']))
    negative_count = min(len(negatives), a.examples//4)
    selected = positives[:a.examples-negative_count] + negatives[:negative_count]
    if len(selected) != a.examples:
        raise ValueError('Insufficient TRAIN examples')
    (a.output/'selected-train.json').write_text(json.dumps(selected, ensure_ascii=False, indent=2)+'\n')
    model, tokenizer = load_components(config, a.bundle/'tokenizer')
    (a.output/'model-loading.json').write_text(json.dumps(model.text2event_load_report, indent=2)+'\n')
    model.to(a.device); model.gradient_checkpointing_enable(); model.config.use_cache = False
    ds = BundleDataset(folder, 'train', tokenizer, config['max_source_length'], config['max_target_length'], source_prefix=config.get('full_source_prefix',''))
    lookup = {r['id']: r for r in ds}
    loader = DataLoader([lookup[g['id']] for g in selected], batch_size=1, shuffle=True, collate_fn=BatchCollator(tokenizer))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.)
    (a.output/'run.json').write_text(json.dumps({'kind':'TRAIN-only-diagnostic-not-baseline', 'dataset':a.dataset,
        'epochs':a.epochs, 'lr':a.lr, 'seed':42, 'config':config},indent=2)+'\n')
    for epoch in range(a.epochs+1):
        metrics = train_epoch(model, loader, opt, scheduler, a.device) if epoch else {}
        if epoch % a.eval_every and epoch != a.epochs:
            continue
        scores, pred, issues = evaluate_model(model, tokenizer, selected, folder/'event.schema', config, a.device)
        metrics.update(epoch=epoch, scores=scores, generation=generation_summary(pred, issues))
        for suffix, data in [('metrics',metrics),('predictions',pred),('diagnostics',issues)]:
            (a.output/f'epoch-{epoch}-{suffix}.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(metrics), flush=True)
    (a.output/'completion.json').write_text(json.dumps({'kind':'diagnostic-not-baseline','epochs':a.epochs,'last_scores':scores})+'\n')


if __name__ == '__main__':
    main()
