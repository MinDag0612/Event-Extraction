"""Derive repetition allowances from TRAIN only; report held-out coverage separately.

Surface-level audit (not a proof that all gold token prefixes pass the decoder).
Does not modify targets, labels, or the input bundle.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path


def required_limits(row):
    tokens = row['tokens']
    # Count distinct original-token spans sharing a surface.
    capacity = Counter(' '.join(tokens[a:b]) for a in range(len(tokens))
                       for b in range(a+1, len(tokens)+1))
    events = Counter(); event_limit = argument_limit = 1
    for event in row['events']:
        for trigger in event['trigger']:
            a, b = trigger['span']; surface = ' '.join(tokens[a:b])
            events[event['event_type'], surface] += 1
            event_limit = max(event_limit, math.ceil(events[event['event_type'], surface]/capacity[surface]))
        arguments = Counter()
        for argument in event['arguments']:
            for mention in argument['mentions']:
                a, b = mention['span']; surface = ' '.join(tokens[a:b])
                arguments[argument['role'], surface] += 1
                argument_limit = max(argument_limit, math.ceil(arguments[argument['role'], surface]/capacity[surface]))
    return event_limit, argument_limit


def audit(bundle):
    result = {}
    for dataset in ['BKEE', 'GENEVA', 'PHEE', 'VHE']:
        folder = bundle/dataset
        path = folder/'train.gold.jsonl'
        train = [required_limits(json.loads(line)) for line in path.read_text().splitlines()]
        event_limit = max(e for e, _ in train)
        argument_limit = max(a for _, a in train)
        record = {'event_repeat_limit':event_limit, 'argument_repeat_limit':argument_limit,
                  'derived_from':'TRAIN only', 'train_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                  'coverage':{}}
        for split in ['train', 'dev', 'test']:
            failures = []; count = 0
            for line in (folder/f'{split}.gold.jsonl').read_text().splitlines():
                row = json.loads(line); count += 1
                e, a = required_limits(row)
                if e > event_limit or a > argument_limit:
                    failures.append({'id':row['id'], 'required_event_limit':e, 'required_argument_limit':a})
            record['coverage'][split] = {'samples':count, 'exceed_limits':len(failures), 'examples':failures[:10]}
        result[dataset] = record
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit(args.bundle),ensure_ascii=False,indent=2)+'\n')
