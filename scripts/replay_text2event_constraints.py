"""Replay gold target token prefixes on supplied diagnostic examples, without model/GPU."""
import argparse
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer
from src.baselines.text2event.convert import convert
from src.baselines.text2event.decoding import TreeConstraint


def replay(bundle, examples, config):
    tokenizer = AutoTokenizer.from_pretrained(bundle/'tokenizer', local_files_only=True)
    results = {}
    for dataset in ['BKEE', 'GENEVA', 'PHEE', 'VHE']:
        rows = json.loads((examples/f'diagnostic-{dataset}'/'selected-train.json').read_text())
        failures = []; passed = 0
        limits = config.get('constraint_repetition_limits', {}).get(dataset, {})
        for row in rows:
            constraint = TreeConstraint(tokenizer, bundle/dataset/'event.schema', [],
                original_tokens=row['tokens'], max_new_tokens=config['generation_max_new_tokens'],
                repetition_policy=config.get('constraint_repetition_policy', 'source-occurrence'),
                event_repeat_limit=limits.get('event_repeat_limit', 1),
                argument_repeat_limit=limits.get('argument_repeat_limit', 1))
            prefix = [tokenizer.pad_token_id]
            for position, token in enumerate(tokenizer.encode(convert(row)['event'])):
                allowed = constraint(0, torch.tensor(prefix))
                if token not in allowed:
                    failures.append({'id':row['id'], 'position':position, 'token':token,
                                     'allowed':allowed[:12]})
                    break
                prefix.append(token)
            else:
                passed += 1
        results[dataset] = {'passed':passed, 'total':len(rows), 'failures':failures}
    return results


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--examples', type=Path, required=True)
    p.add_argument('--config', type=Path, default=Path('configs/text2event/preparation.json'))
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = replay(args.bundle, args.examples, json.loads(args.config.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(result, ensure_ascii=False))
    if any(row['failures'] for row in result.values()):
        raise SystemExit(1)
